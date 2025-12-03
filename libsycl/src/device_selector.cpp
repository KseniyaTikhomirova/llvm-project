//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/device_selector.hpp>

#include <detail/device_impl.hpp>

#include <algorithm>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

static constexpr int MatchedTypeDefaultScore = 1000;
static constexpr int GPUDeviceDefaultScore = 500;
static constexpr int CPUDeviceDefaultScore = 300;
static constexpr int AccDeviceDefaultScore = 75;
static constexpr int RejectDeviceScore = -1;

static int getDevicePreference(const device &Device) {
  int Score = 0;
  const detail::device_impl &DeviceImpl = getSyclObjImpl(Device);

  // TODO: increase score for devices with compatible dev images

  if (DeviceImpl.getBackend() == backend::level_zero)
    Score += 50;

  return Score;
}

_LIBSYCL_EXPORT int default_selector_v(const device &dev) {
  int Score = 0;

  if (dev.is_gpu())
    Score += GPUDeviceDefaultScore;

  if (dev.is_cpu())
    Score += CPUDeviceDefaultScore;

  if (dev.is_accelerator())
    Score += AccDeviceDefaultScore;

  Score += getDevicePreference(dev);

  return Score;
}

_LIBSYCL_EXPORT int gpu_selector_v(const device &dev) {
  return dev.is_gpu() ? MatchedTypeDefaultScore + getDevicePreference(dev)
                      : RejectDeviceScore;
}

_LIBSYCL_EXPORT int cpu_selector_v(const device &dev) {
  return dev.is_cpu() ? MatchedTypeDefaultScore + getDevicePreference(dev)
                      : RejectDeviceScore;
}

_LIBSYCL_EXPORT int accelerator_selector_v(const device &dev) {
  return dev.is_accelerator()
             ? MatchedTypeDefaultScore + getDevicePreference(dev)
             : RejectDeviceScore;
}

_LIBSYCL_EXPORT detail::DeviceSelectorInvocableType
aspect_selector(const std::vector<aspect> &RequireList,
                const std::vector<aspect> &DenyList) {
  return [=](const sycl::device &Dev) {
    // 4.6.1.1. Device selector:
    // If no aspects are passed in, the generated selector behaves like
    // default_selector_v.
    if (RequireList.empty() && DenyList.empty())
      return default_selector_v(Dev);

    auto HasAspect = [&Dev](const aspect &Aspect) -> bool {
      return Dev.has(Aspect);
    };
    if (!std::all_of(RequireList.begin(), RequireList.end(), HasAspect))
      return RejectDeviceScore;

    if (std::any_of(DenyList.begin(), DenyList.end(), HasAspect))
      return RejectDeviceScore;

    return MatchedTypeDefaultScore + getDevicePreference(Dev);
  };
}

namespace detail {

_LIBSYCL_EXPORT device
select_device(const DeviceSelectorInvocableType &DeviceSelector) {
  int ChosenDeviceScore = RejectDeviceScore;
  const device *ChosenDevice = nullptr;

  std::vector<device> Devices = device::get_devices();
  for (const auto &Device : Devices) {
    int CurrentDevScore = DeviceSelector(Device);
    if (CurrentDevScore < 0)
      continue;

    if ((ChosenDeviceScore < CurrentDevScore) ||
        ((ChosenDeviceScore == CurrentDevScore) &&
         (getDevicePreference(*ChosenDevice) < getDevicePreference(Device)))) {
      ChosenDevice = &Device;
      ChosenDeviceScore = CurrentDevScore;
    }
  }

  if (ChosenDevice != nullptr) {
    return *ChosenDevice;
  }

  auto Selector = DeviceSelector.target<int (*)(const sycl::device &)>();
  assert(Selector &&
         "Provided device selector violates SYCL 2020 requirements: must be "
         "Callable, taking a parameter of type const device & and returning a "
         "value that is implicitly convertible to int.");

#define DEFAULT_MESSAGE "No device of requested type is available"
  constexpr const char DefaultMessage[] = DEFAULT_MESSAGE;
  constexpr const char CpuMessage[] =
      DEFAULT_MESSAGE ": 'info::device_type::cpu' ";
  constexpr const char GpuMessage[] =
      DEFAULT_MESSAGE ": 'info::device_type::gpu' ";
  constexpr const char AccMessage[] =
      DEFAULT_MESSAGE ": 'info::device_type::accelerator' ";
#undef DEFAULT_MESSAGE
  std::error_code Errc = make_error_code(errc::runtime);

  if (*Selector == gpu_selector_v)
    throw exception(Errc, GpuMessage);
  else if (*Selector == cpu_selector_v)
    throw exception(Errc, CpuMessage);
  else if (*Selector == accelerator_selector_v)
    throw exception(Errc, AccMessage);

  throw exception(Errc, DefaultMessage);
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
