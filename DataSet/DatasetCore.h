#pragma once

#define namespace_dataset_start namespace mogi { namespace dataset {
#define namespace_dataset_end } }

// For the dll build define EXPORT_LIBRARY
#ifdef EXPORT_DATASET_LIBRARY
#define DATASET_API __declspec(dllexport)
#else
#define DATASET_API __declspec(dllimport)
#endif // EXPORT_LIBRARY