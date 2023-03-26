#include "logger.h"

namespace SimpleInfer {

void InitializeLogger() {
    static bool has_initialized = false;
    if (!has_initialized) {
        absl::InitializeLog();
        SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
        has_initialized = true;
    }
}

}  // namespace SimpleInfer
