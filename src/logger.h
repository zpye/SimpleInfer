#ifndef SIMPLE_INFER_SRC_LOGGER_H_
#define SIMPLE_INFER_SRC_LOGGER_H_

#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

namespace SimpleInfer {

void InitializeLogger();

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LOGGER_H_
