// NPU sensors: per-column utilisation, power and temperature, via the amdxdna
// DRM ioctl.
//
// WHY THIS EXISTS. The obvious ways to ask "how busy is the NPU" all come up empty
// on this stack, and the reasons are worth recording so nobody re-derives them:
//
//   * hwmon exposes only TWO channels for the NPU: `power1_input` (NPU_power) and
//     `temp1_input` (NPU_temperature). The per-column busy counters exist in the
//     same PMF read but are NOT wired into hwmon.
//   * fdinfo reports `drm-engine-<driver name>`, i.e. per-FILE accumulated submit
//     time, and it is per `drm_file` -- so it is per client, not per column, and
//     only moves for the client that owns the hardware context.
//   * amd-smi does not report the AIE at all (it reports GFX/VCN/IPU).
//
// The counter IS reachable: `amdxdna_query_sensors()` fills a caller-provided
// buffer with `amdxdna_drm_query_sensor` records, one per metric, including one
// `AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION` per AIE column in PERCENT. That is the
// NPU's own busy signal, and nothing in the tree was reading it.
//
// UNITS. Each record carries its own `units` string and a `unitm` exponent, and the
// driver sets them per metric: power is mW with unitm=-3, temperature is C with
// unitm=0, column utilisation is % with unitm=0. So print `units` and apply
// `pow(10, unitm)` rather than assuming watts -- the hwmon view of the same power
// is in microwatts, which is a 1000x difference from this ioctl's mW, and mixing
// the two is how you get a confident, wrong number.
//
// The two-call size protocol: pass buffer_size=0 and the kernel writes the size it
// needs, then allocate and call again. Do not guess the record count.
//
// build:
//   g++ -std=c++17 -O2 -o /tmp/npu-sensors npu-sensors.cpp \
//       -I<repo>/include/uapi
#include <cerrno>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

extern "C" {
#include <drm/amdxdna_accel.h>
}

namespace {

const char *type_name(unsigned t) {
  switch (t) {
    case AMDXDNA_SENSOR_TYPE_POWER: return "power";
    case AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION: return "column_util";
    // There is deliberately NO AMDXDNA_SENSOR_TYPE_TEMPERATURE case.
    //
    // The installed uapi header (/usr/include/drm/amdxdna_accel.h) defines only
    // POWER and COLUMN_UTILIZATION, so naming TEMPERATURE here did not compile --
    // this file had never been built against the installed header. It matches
    // the hardware: amdxdna_sensors.c:75 skips the temperature record unless
    // HAVE_7_2_AMD_PMF_NPU_METRICS_NPU_TEMP is defined, which this kernel does
    // not, and hwmon temp1_input correspondingly reads EOPNOTSUPP.
    default: return "?";
  }
}

}  // namespace

int main(int argc, char **argv) {
  const char *node = (argc > 1) ? argv[1] : "/dev/accel/accel0";
  int quiet = (argc > 2 && !strcmp(argv[2], "-q"));

  int fd = open(node, O_RDWR);
  if (fd < 0) {
    fprintf(stderr, "open %s: %s\n", node, strerror(errno));
    return 1;
  }

  // ONE call, with a full-size buffer.
  //
  // There is NO size-query pass. `amdxdna_query_sensors()` does
  // `if (args->buffer_size < sizeof(sensor)) goto out;` per record and then, at
  // `out:`, sets `args->buffer_size = sensors_count * sizeof(sensor)`. So a
  // size-probe with buffer_size == 0 skips every record, counts zero, and
  // returns 0 records AND rc 0 -- which is exactly the "0 records, no error"
  // this tool used to report as "PMF sensors unavailable".
  //
  // Maximum records = 1 power + AMDXDNA_NPU_MAX_PMF_COLUMNS (0x8) column
  // utilization + 1 temperature. Passing 16 is generous and costs nothing; the
  // driver returns however many it actually has and writes the real count back
  // into buffer_size.
  const unsigned kMaxRec = 16;
  void *buf = calloc(kMaxRec, sizeof(amdxdna_drm_query_sensor));
  if (!buf) { perror("calloc"); close(fd); return 1; }

  amdxdna_drm_get_info info{};
  info.param = DRM_AMDXDNA_QUERY_SENSORS;
  info.buffer_size = kMaxRec * sizeof(amdxdna_drm_query_sensor);
  info.buffer = (unsigned long long)(unsigned long)buf;
  if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info) < 0) {
    fprintf(stderr, "GET_INFO(sensors) on %s: %s\n", node, strerror(errno));
    fprintf(stderr,
            "If this is EPERM/EINVAL the accel node may need a render-group "
            "hand-off; the NPU node is normally /dev/accel/accelN owned by group "
            "'render'.\n");
    free(buf); close(fd); return 1;
  }

  const unsigned got = info.buffer_size / sizeof(amdxdna_drm_query_sensor);
  if (!quiet) {
    printf("node %s  records %u/%u\n", node, got, kMaxRec);
  }
  for (unsigned i = 0; i < got; ++i) {
    const auto &s = static_cast<const amdxdna_drm_query_sensor *>(buf)[i];
    // The label is NOT NUL-terminated by the driver (scnprintf into a fixed
    // buffer), so bound it explicitly. Printing it with %s would run into
    // whatever followed in the record.
    char label[sizeof(s.label) + 1] = {0};
    memcpy(label, s.label, sizeof(s.label));
    if (quiet) {
      printf("%s %s %u%s\n", type_name(s.type), label, s.input, s.units);
    } else {
      printf("  %-14s %-24s %8u %-3s (unitm %d, max %u, avg %u, high %u)\n",
             type_name(s.type), label, s.input, s.units, (int)s.unitm, s.max,
             s.average, s.highest);
    }
  }

  free(buf);
  close(fd);
  return 0;
}
