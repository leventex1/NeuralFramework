#pragma once

#define namespace_accelerator_start namespace mogi { namespace accelerator {
#define namespace_accelerator_end } }

// For the dll build define EXPORT_LIBRARY
#ifdef EXPORT_ACCELERATOR_LIBRARY
#define LIBRARY_ACCELERATOR_API __declspec(dllexport)
#else
#define LIBRARY_ACCELERATOR_API __declspec(dllimport)
#endif // EXPORT_LIBRARY
