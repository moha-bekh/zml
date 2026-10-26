//! Explicit hardware regression: SM75 must select the portable attention path.
const std = @import("std");
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .warn };

const queries = 1;
const keys = 64;
const heads = 8;
const head_dim = 64;

fn fill(values: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| value.* = random.float(f32) - 0.5;
}

pub fn main(init: std.process.Init) !void {
    const platform = try zml.Platform.init(init.gpa, init.io, .cuda, .{});
    defer platform.deinit(init.gpa, init.io);
    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnknownComputeCapability;
    if (cc.major != 7 or cc.minor != 5) return error.RequiresSM75;
    const automatic = zml.attention.Backend.auto(platform);
    if (automatic != .stablehlo) return error.UnexpectedAttentionBackend;

    var q_values: [queries * heads * head_dim]f32 = undefined;
    var k_values: [keys * heads * head_dim]f32 = undefined;
    var v_values: [keys * heads * head_dim]f32 = undefined;
    fill(&q_values, 17);
    fill(&k_values, 29);
    fill(&v_values, 43);
    const positions = [_]u32{ 0, 31, 63 };
    const q_shape: zml.Shape = .init(.{ .q = queries, .h = heads, .hd = head_dim }, .f32);
    const k_shape: zml.Shape = .init(.{ .k = keys, .h = heads, .hd = head_dim }, .f32);
    const position_shape: zml.Shape = .init(.{}, .u32);
    var q = try zml.Buffer.fromBytes(init.io, platform, q_shape, std.mem.sliceAsBytes(&q_values));
    defer q.deinit();
    var k = try zml.Buffer.fromBytes(init.io, platform, k_shape, std.mem.sliceAsBytes(&k_values));
    defer k.deinit();
    var v = try zml.Buffer.fromBytes(init.io, platform, k_shape, std.mem.sliceAsBytes(&v_values));
    defer v.deinit();

    const backends = [_]zml.attention.Backend{ .{ .stablehlo = {} }, automatic };
    for (backends, 0..) |backend, index| {
        const metadata: zml.attention.Metadata = .init(.fromBackend(backend));
        const parameters: zml.attention.Parameters = .init(.fromBackend(backend));
        var exe = try platform.compileFn(init.gpa, init.io, zml.attention.attention, .{
            zml.Tensor.fromShape(q_shape), zml.Tensor.fromShape(k_shape),
            zml.Tensor.fromShape(k_shape), zml.Tensor.fromShape(position_shape),
            metadata,                      parameters,
        }, .{});
        defer exe.deinit();
        var metadata_buffer = try metadata.initBuffer(init.io, platform);
        defer zml.attention.Metadata.deinitBuffer(&metadata_buffer);
        for (positions) |position| {
            var token_index = try zml.Buffer.scalar(init.io, platform, position, .u32);
            defer token_index.deinit();
            var result = try exe.eval(init.gpa, init.io, .{ q, k, v, token_index, metadata_buffer });
            defer result.deinit();
            const actual = try result.toSliceAlloc(init.gpa, init.io);
            defer actual.free(init.gpa);
            try checkReference(&q_values, &k_values, &v_values, &.{position}, actual.items(f32));
        }
        std.debug.print("SM75 attention: path={s}, backend={s}, three positions passed f64 oracle\n", .{
            if (index == 0) "forced" else "automatic", @tagName(backend),
        });
    }
}

fn checkReference(q: []const f32, k: []const f32, v: []const f32, positions: []const u32, actual: []const f32) !void {
    if (actual.len != queries * heads * head_dim) return error.UnexpectedOutputShape;
    for (positions, 0..) |position, query| {
        for (0..heads) |head| {
            var scores: [keys]f64 = undefined;
            var maximum = -std.math.inf(f64);
            for (0..position + 1) |key| {
                var dot: f64 = 0;
                for (0..head_dim) |dim| {
                    dot += @as(f64, q[(query * heads + head) * head_dim + dim]) *
                        @as(f64, k[(key * heads + head) * head_dim + dim]);
                }
                scores[key] = dot / @sqrt(@as(f64, head_dim));
                maximum = @max(maximum, scores[key]);
            }
            var denominator: f64 = 0;
            for (scores[0 .. position + 1]) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            for (0..head_dim) |dim| {
                var expected: f64 = 0;
                for (scores[0 .. position + 1], 0..) |weight, key| {
                    expected += weight * @as(f64, v[(key * heads + head) * head_dim + dim]);
                }
                expected /= denominator;
                const got: f64 = actual[(query * heads + head) * head_dim + dim];
                if (!(@abs(got - expected) <= 2e-5 + 2e-5 * @abs(expected))) {
                    std.debug.print("attention mismatch: query={d}, head={d}, dim={d}, expected={d}, actual={d}\n", .{
                        query, head, dim, expected, got,
                    });
                    return error.AttentionMismatch;
                }
            }
        }
    }
}
