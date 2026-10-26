//! Numerical regression for independent GatedDeltaNet layer states; no checkpoint required.
const std = @import("std");
const zml = @import("zml");
const model = @import("qwen3_5/model.zig");

pub const std_options: std.Options = .{ .log_level = .warn };
const Cache = model.KvCache.GatedDeltaNetCache;
const State = struct { conv: zml.Buffer, recurrent: zml.Buffer };
const Output = struct { hidden: zml.Tensor, conv: zml.Tensor, recurrent: zml.Tensor };

fn forward(layer: model.GatedDeltaNet, x: zml.Tensor, conv: zml.Tensor, recurrent: zml.Tensor, index: zml.Tensor, active: zml.Tensor) Output {
    const y, const cache = layer.forward(x, .{ .conv_state = conv, .recurrent_state = recurrent, .layer_index = index }, active);
    return .{ .hidden = y, .conv = cache.conv_state, .recurrent = cache.recurrent_state };
}

fn config() model.Config {
    return .{ .text_config = .{
        .num_hidden_layers = 3,
        .layer_types = &.{ .linear_attention, .linear_attention, .linear_attention },
        .hidden_size = 8,
        .max_position_embeddings = 128,
        .rms_norm_eps = 1e-5,
        .head_dim = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .rope_parameters = .{ .mrope_section = .{ 1, 0, 0 }, .partial_rotary_factor = 1, .rope_theta = 10000 },
        .linear_conv_kernel_dim = 4,
        .linear_key_head_dim = 2,
        .linear_num_key_heads = 2,
        .linear_num_value_heads = 4,
        .linear_value_head_dim = 2,
    } };
}

fn makeLayer(mesh: *const zml.Mesh) model.GatedDeltaNet {
    const Proj = struct {
        fn make(out: i64, mesh_: *const zml.Mesh) zml.nn.Linear {
            return .init(.fromShape(zml.Shape.init(.{ .dout = out, .d = 8 }, .f32).withPartitioning(mesh_, .{ .dout = .model })), null, .d);
        }
    };
    return .{
        .in_proj_qkv = Proj.make(16, mesh),
        .in_proj_z = Proj.make(8, mesh),
        .in_proj_b = Proj.make(4, mesh),
        .in_proj_a = Proj.make(4, mesh),
        .out_proj = .init(.fromShape(zml.Shape.init(.{ .dout = 8, .d = 8 }, .f32).withPartitioning(mesh, .{ .d = .model })), null, .d),
        .conv1d_weight = .fromShape(zml.Shape.init(.{ .out = 16, .in = 1, .kernel_size = 4 }, .f32).withPartitioning(mesh, .{ .out = .model })),
        .dt_bias = .fromShape(zml.Shape.init(.{ .vh = 4 }, .f32).withPartitioning(mesh, .{ .vh = .model })),
        .aLog = .fromShape(zml.Shape.init(.{ .vh = 4 }, .f32).withPartitioning(mesh, .{ .vh = .model })),
        .norm = .{ .weight = .init(.{ .d = 2 }, .f32), .eps = 1e-5 },
        .num_k_heads = 2,
        .num_v_heads = 4,
        .qk_head_repetition = 2,
        .head_k_dim = 2,
        .head_v_dim = 2,
        .conv_kernel_size = 4,
    };
}

const Fill = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    seed: usize = 0,
    fn buffer(ctx: *Fill, tensor: zml.Tensor) zml.Buffer {
        var slice = zml.Slice.alloc(ctx.allocator, tensor.shape()) catch @panic("test allocation failed");
        defer slice.free(ctx.allocator);
        for (slice.items(f32), 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt((i + ctx.seed) % 13)) / 100;
        ctx.seed += 3;
        return zml.Buffer.fromSlice(ctx.io, ctx.platform, slice) catch @panic("test buffer upload failed");
    }
};

pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, struct { cuda: bool = false });
    if (args.cuda) {
        const platform = try zml.Platform.init(init.gpa, init.io, .cuda, .{});
        defer platform.deinit(init.gpa, init.io);
        const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
        try check(init.gpa, init.io, platform, mesh);
        std.debug.print("GatedDeltaNet states match on CUDA\n", .{});
        return;
    }
    for ([_]u32{ 1, 2, 4 }) |devices| {
        const platform = try zml.Platform.init(init.gpa, init.io, .cpu, .{ .cpu = .{ .device_count = devices } });
        defer platform.deinit(init.gpa, init.io);
        const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
        if (mesh.numPartitionsForLogicalAxis(.model) != devices) return error.UnexpectedPartitions;
        try check(init.gpa, init.io, platform, mesh);
        std.debug.print("GatedDeltaNet states match: {d} partitions, 3 layers, full/padded prefill + 32 decode steps\n", .{devices});
    }
}

fn check(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh) !void {
    const full = Cache.init(config(), 1, .f32, .f32, mesh);
    const single = full.singleLayer();
    const mdl = makeLayer(mesh);
    var fill: Fill = .{ .allocator = allocator, .io = io, .platform = platform };
    var weights: zml.Bufferized(model.GatedDeltaNet) = undefined;
    try zml.meta.mapAlloc(Fill.buffer, allocator, &fill, mdl, &weights);
    defer zml.Buffer.deinitAll(model.GatedDeltaNet, &weights);
    var conv = try zml.Slice.alloc(allocator, full.conv_state.shape());
    defer conv.free(allocator);
    var recurrent = try zml.Slice.alloc(allocator, full.recurrent_state.shape());
    defer recurrent.free(allocator);
    for (conv.items(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 1000;
    for (recurrent.items(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 1000;
    var global: State = .{ .conv = try .fromSlice(io, platform, conv), .recurrent = try .fromSlice(io, platform, recurrent) };
    defer global.conv.deinit();
    defer global.recurrent.deinit();
    var separate: [3]State = undefined;
    for (&separate, 0..) |*state, index| {
        const cb = single.conv_state.shape().byteSize();
        const rb = single.recurrent_state.shape().byteSize();
        state.* = .{
            .conv = try .fromBytes(io, platform, single.conv_state.shape(), conv.bytes[index * cb ..][0..cb]),
            .recurrent = try .fromBytes(io, platform, single.recurrent_state.shape(), recurrent.bytes[index * rb ..][0..rb]),
        };
    }
    defer for (&separate) |*state| {
        state.conv.deinit();
        state.recurrent.deinit();
    };
    const active_tensor: zml.Tensor = .init(.{}, .u32);
    const index_tensor: zml.Tensor = .init(.{}, .u32);
    // Compile both layouts for prefill and decoding. Every layer gets distinct input values.
    inline for (.{ @as(i64, 4), @as(i64, 1), @as(i64, 32), @as(i64, 1) }) |seqlen| {
        const x: zml.Tensor = .init(.{ .b = 1, .s = seqlen, .d = 8 }, .f32);
        var full_exe = try platform.compileFn(allocator, io, forward, .{ mdl, x, full.conv_state, full.recurrent_state, index_tensor, active_tensor }, .{ .meshes = &.{mesh} });
        defer full_exe.deinit();
        var single_exe = try platform.compileFn(allocator, io, forward, .{ mdl, x, single.conv_state, single.recurrent_state, index_tensor, active_tensor }, .{ .meshes = &.{mesh} });
        defer single_exe.deinit();
        var full_runner = try full_exe.runner(allocator);
        defer full_runner.deinit(allocator);
        var single_runner = try single_exe.runner(allocator);
        defer single_runner.deinit(allocator);
        var active = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(if (seqlen == 32) 17 else seqlen)), .u32);
        defer active.deinit();
        var zero = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
        defer zero.deinit();
        for (0..(if (seqlen == 1) @as(usize, 16) else 1)) |step| {
            for (&separate, 0..) |*state, index| {
                var input = Fill.buffer(&fill, x);
                defer input.deinit();
                var global_index = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(index)), .u32);
                defer global_index.deinit();
                var global_y: zml.Buffer = undefined;
                full_runner.run(io, .{ weights, input, global.conv, global.recurrent, global_index, active }, .{ .hidden = &global_y, .conv = &global.conv, .recurrent = &global.recurrent }, .{ .wait = true });
                defer global_y.deinit();
                var single_y: zml.Buffer = undefined;
                single_runner.run(io, .{ weights, input, state.conv, state.recurrent, zero, active }, .{ .hidden = &single_y, .conv = &state.conv, .recurrent = &state.recurrent }, .{ .wait = true });
                defer single_y.deinit();
                try zml.testing.expectClose(io, global_y, single_y, .{ .absolute_tolerance = 1e-6, .relative_tolerance = 1e-5 });
                // Compare every layer after each update, including layers not touched by this call.
                inline for (.{ "conv", "recurrent" }) |field| {
                    const all = try @field(global, field).toSliceAlloc(allocator, io);
                    defer all.free(allocator);
                    for (separate, 0..) |other, other_index| {
                        const actual = try @field(other, field).toSliceAlloc(allocator, io);
                        defer actual.free(allocator);
                        const bytes = actual.bytes.len;
                        const expected = zml.Slice.init(actual.shape, all.bytes[other_index * bytes ..][0..bytes]);
                        try zml.testing.expectClose(io, expected, actual, .{ .absolute_tolerance = 1e-6, .relative_tolerance = 1e-5 });
                    }
                }
                _ = step;
            }
        }
    }
}
