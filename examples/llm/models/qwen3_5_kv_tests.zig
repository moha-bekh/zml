//! Compile small attention graphs without downloading model weights.
const std = @import("std");
const zml = @import("zml");

const dense = @import("qwen3_5/model.zig");
const moe = @import("qwen3_5_moe/model.zig");

pub const std_options: std.Options = .{ .log_level = .warn };

pub fn main(init: std.process.Init) !void {
    for ([_]u32{ 1, 2, 4 }) |device_count| {
        const platform = try zml.Platform.init(init.gpa, init.io, .cpu, .{
            .cpu = .{ .device_count = device_count },
        });
        defer platform.deinit(init.gpa, init.io);
        const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
        if (mesh.numPartitionsForLogicalAxis(.model) != device_count) return error.UnexpectedPartitionCount;

        inline for (.{ dense, moe }) |model| {
            try compileAttention(model, init.gpa, init.io, platform, mesh, 2);
            if (device_count == 4) try compileAttention(model, init.gpa, init.io, platform, mesh, 4);
        }
    }
}

fn compileAttention(
    comptime model: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mesh: *const zml.Mesh,
    kv_heads: i64,
) !void {
    var self: model.SelfAttn = undefined;
    self.q_proj = .init(.init(.{ .dout = 128, .d = 16 }, .f32), null, .d);
    self.k_proj = .init(.init(.{ .dout = kv_heads * 8, .d = 16 }, .f32), null, .d);
    self.v_proj = .init(.init(.{ .dout = kv_heads * 8, .d = 16 }, .f32), null, .d);
    self.o_proj = .init(.init(.{ .dout = 16, .d = 64 }, .f32), null, .d);
    self.q_norm = .{ .weight = .init(.{ .d = 8 }, .f32) };
    self.k_norm = .{ .weight = .init(.{ .d = 8 }, .f32) };
    self.num_heads = 8;
    self.num_kv_heads = kv_heads;
    self.head_dim = 8;
    self.rotary_dim = 4;
    self.rotary_embed = .init(4, 10_000, .{ 1, 1, 0 });
    inline for (.{ "q_proj_scale", "k_proj_scale", "v_proj_scale", "o_proj_scale" }) |name| {
        if (@hasField(model.SelfAttn, name)) @field(self, name) = null;
    }
    const sharding = mesh.shardableDim(kv_heads, .model, self.num_heads);
    const cache_heads = switch (sharding) {
        .sharded => |heads| heads.dim,
        .replicated => kv_heads,
    };
    const cache_shape = zml.Shape.init(.{
        .b = 1,
        .layer = 1,
        .s = 4,
        .h = cache_heads,
        .hd = 8,
    }, .f32).withPartitioning(mesh, .{ .h = .model });
    const cache: model.KvCache.SelfAttnCache = .{
        .k = .fromShape(cache_shape),
        .v = .fromShape(cache_shape),
        .layer_index = .init(.{}, .u32),
    };
    const x: zml.Tensor = .init(.{ .b = 1, .s = 1, .d = 16 }, .f32);
    const token_index: zml.Tensor = .init(.{}, .u32);

    // The regression aborts in cache.update when two heads are replicated twice
    // across four partitions: the update has eight heads but the cache has four.
    var exe = try platform.compileFn(allocator, io, model.SelfAttn.forward, .{
        self, x, token_index, cache,
    }, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    std.debug.print("Qwen KV graph compiled: model={s}, partitions={d}, KV heads={d}, cache heads={d}\n", .{
        @typeName(model), mesh.numPartitionsForLogicalAxis(.model), kv_heads, cache_heads,
    });
}
