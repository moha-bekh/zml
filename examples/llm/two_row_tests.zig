//! CPU BF16 two-row regression against original dot and a scalar f64 oracle.
const std = @import("std");
const zml = @import("zml");
const BF = zml.floats.BFloat16;
pub const std_options: std.Options = .{ .log_level = .warn };

fn original(w: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(w, .d).convert(.bf16).transpose(.{ .s, .dout });
}

pub fn main(init: std.process.Init) !void {
    const platform = try zml.Platform.init(init.gpa, init.io, .cpu, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(init.gpa, init.io);
    inline for (.{ .{ 1, 16, 32 }, .{ 1, 1024, 128 }, .{ 1, 1024, 4096 }, .{ 8, 1024, 128 }, .{ 64, 1024, 128 } }) |dims| {
        inline for (.{ false, true }) |with_bias| try check(init.gpa, init.io, platform, dims[0], dims[1], dims[2], with_bias);
    }
}

fn check(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, rows: i64, inner: i64, cols: i64, comptime with_bias: bool) !void {
    const w: zml.Tensor = .init(.{ .dout = cols, .d = inner }, .bf16);
    const x: zml.Tensor = .init(.{ .s = rows, .d = inner }, .bf16);
    const Functions = struct {
        fn old(wt: zml.Tensor, xt: zml.Tensor) zml.Tensor {
            const y = original(wt, xt);
            return if (with_bias) y.add(zml.Tensor.scalar(0.03125, .bf16).broad(y.shape())) else y;
        }
        fn new(wt: zml.Tensor, xt: zml.Tensor) zml.Tensor {
            const bias: ?zml.Tensor = if (with_bias) zml.Tensor.scalar(0.03125, .bf16).broad(wt.shape().drop(.d)) else null;
            return zml.nn.Linear.init(wt, bias, .d).forward(xt, .bf16).transpose(.{ .s, .dout });
        }
    };
    var old_exe = try platform.compileFn(allocator, io, Functions.old, .{ w, x }, .{});
    defer old_exe.deinit();
    var new_exe = try platform.compileFn(allocator, io, Functions.new, .{ w, x }, .{});
    defer new_exe.deinit();
    var old_runner = try old_exe.runner(allocator);
    defer old_runner.deinit(allocator);
    var new_runner = try new_exe.runner(allocator);
    defer new_runner.deinit(allocator);
    const k: usize = @intCast(inner);
    const n: usize = @intCast(cols);
    const m: usize = @intCast(rows);
    const ws = try allocator.alloc(BF, n * k);
    defer allocator.free(ws);
    const expanded = try allocator.alloc(f32, ws.len);
    defer allocator.free(expanded);
    const xs = try allocator.alloc(BF, m * k);
    defer allocator.free(xs);
    for (0..3) |seed| {
        var prng: std.Random.DefaultPrng = .init(0x13209 + seed);
        const random = prng.random();
        for (ws, expanded) |*v, *f| {
            v.* = BF.fromF32((random.float(f32) - 0.5) * 0.2);
            f.* = v.toF32();
        }
        for (xs) |*v| v.* = BF.fromF32((random.float(f32) - 0.5) * 0.2);
        var wb = try zml.Buffer.fromBytes(io, platform, w.shape(), std.mem.sliceAsBytes(ws));
        defer wb.deinit();
        var xb = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(xs));
        defer xb.deinit();
        var old_y: zml.Buffer = undefined;
        old_runner.run(io, .{ wb, xb }, .{&old_y}, .{ .wait = true });
        defer old_y.deinit();
        var new_y: zml.Buffer = undefined;
        new_runner.run(io, .{ wb, xb }, .{&new_y}, .{ .wait = true });
        defer new_y.deinit();
        const old_slice = try old_y.toSliceAlloc(allocator, io);
        defer old_slice.free(allocator);
        const new_slice = try new_y.toSliceAlloc(allocator, io);
        defer new_slice.free(allocator);
        const before = old_slice.items(BF);
        const after = new_slice.items(BF);
        var different: usize = 0;
        var max_delta: f32 = 0;
        var max_oracle_error: f32 = 0;
        for (0..m) |row| {
            var best_reference: usize = 0;
            var best_before: usize = 0;
            var best_after: usize = 0;
            var best_value: f32 = -std.math.inf(f32);
            for (0..n) |col| {
                var sum: f64 = 0;
                for (0..k) |j| sum += @as(f64, xs[row * k + j].toF32()) * @as(f64, expanded[col * k + j]);
                const truth = BF.fromF32(BF.fromF32(@floatCast(sum)).toF32() + (if (with_bias) @as(f32, 0.03125) else 0)).toF32();
                const a = before[row * n + col].toF32();
                const b = after[row * n + col].toF32();
                if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonFinite;
                if (@as(u16, @bitCast(before[row * n + col])) != @as(u16, @bitCast(after[row * n + col]))) different += 1;
                max_delta = @max(max_delta, @abs(a - b));
                max_oracle_error = @max(max_oracle_error, @max(@abs(a - truth), @abs(b - truth)));
                // Allow BF16 rounding and a different FP32 reduction order.
                const tolerance = 1e-5 + 1e-2 * @abs(truth);
                if (@abs(a - truth) > tolerance or @abs(b - truth) > tolerance) return error.OracleMismatch;
                if (truth > best_value) {
                    best_value = truth;
                    best_reference = col;
                }
                if (a > before[row * n + best_before].toF32()) best_before = col;
                if (b > after[row * n + best_after].toF32()) best_after = col;
            }
            if (best_before != best_after or best_after != best_reference) return error.ArgmaxMismatch;
        }
        std.debug.print("PRECISION bias={any} rows={d} inner={d} cols={d} seed={d} bit_differences={d} max_delta={e} max_oracle_error={e} argmax=equal\n", .{ with_bias, rows, inner, cols, seed, different, max_delta, max_oracle_error });
    }
}
