const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo) !void {
    const info = @typeInfo(DeviceInfo).@"union";
    inline for (info.field_names, info.field_types) |type_name, DeviceType| {
        const tag = @field(smi_info.Target, type_name);
        var header_printed = false;

        for (devices, 0..) |dev, i| {
            switch (dev.*) {
                tag => |*db| {
                    if (!header_printed) {
                        try writer.writeAll("index,type");
                        inline for (@typeInfo(DeviceType.Value).@"struct".field_names) |field_name| {
                            try writer.writeAll("," ++ field_name);
                        }
                        try writer.writeAll("\n");

                        header_printed = true;
                    }

                    const val = db.front().*;

                    try writer.print("{d},{s}", .{ i, type_name });
                    inline for (@typeInfo(DeviceType.Value).@"struct".field_names) |field_name| {
                        try writer.writeAll(",");
                        try writeValue(writer, @field(val, field_name));
                    }
                    try writer.writeAll("\n");
                },
                else => {},
            }
        }
    }
}

fn writeValue(writer: *std.Io.Writer, field: anytype) !void {
    if (@typeInfo(@TypeOf(field)) == .optional) {
        if (field) |v| return writeValue(writer, v);
    } else switch (@typeInfo(@TypeOf(field))) {
        .pointer => {
            if (std.mem.indexOfAny(u8, field, ",\"\r\n") == null) return writer.writeAll(field);
            try writer.writeByte('"');
            for (field) |byte| {
                if (byte == '"') try writer.writeByte('"');
                try writer.writeByte(byte);
            }
            return writer.writeByte('"');
        },
        else => return writer.print("{any}", .{field}),
    }
}

test "CSV strings escape commas quotes and line breaks" {
    const cases = .{
        .{ "plain", "plain" },
        .{ "", "" },
        .{ "GPU, A", "\"GPU, A\"" },
        .{ "GPU \"A\"", "\"GPU \"\"A\"\"\"" },
        .{ "GPU\rA", "\"GPU\rA\"" },
        .{ "GPU\nA", "\"GPU\nA\"" },
        .{ "GPU\r\nA", "\"GPU\r\nA\"" },
        .{ "GPU, quoted \"name\"\n", "\"GPU, quoted \"\"name\"\"\n\"" },
    };
    inline for (cases) |case| {
        var writer = std.Io.Writer.Allocating.init(std.testing.allocator);
        defer writer.deinit();
        try writeValue(&writer.writer, @as([]const u8, case[0]));
        try std.testing.expectEqualStrings(case[1], writer.written());
    }
}

test "CSV device names are escaped through optional fields" {
    var dev: DeviceInfo = .{ .cuda = .{ .values = .{ .{ .name = "GPU, quoted \"name\"", .util_percent = 0 }, .{} } } };
    var writer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer writer.deinit();
    try write(&writer.writer, &.{&dev});
    try std.testing.expect(std.mem.indexOf(u8, writer.written(), ",\"GPU, quoted \"\"name\"\"\",") != null);
}
