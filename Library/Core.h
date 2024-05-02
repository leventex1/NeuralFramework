#pragma once

#define ASYNC 1

#define namespace_start namespace mogi {
#define namespace_end }

// For the dll build define EXPORT_LIBRARY
#ifdef EXPORT_LIBRARY
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif // EXPORT_LIBRARY
