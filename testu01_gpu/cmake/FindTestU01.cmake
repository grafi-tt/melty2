# FindTestU01: the headers and the libraries probdist and mylib of TestU01 (TestU01_ROOT: its installation
# prefix). Defines TestU01_FOUND, TestU01_INCLUDE_DIR and the imported targets TestU01::probdist (with the
# include directory) and TestU01::mylib. Besides the layout of TestU01 itself, that of the packages of Debian and
# Ubuntu (libtestu01-0-dev) is found: the headers in include/testu01, the libraries testu01probdist and testu01mylib.
find_path(TestU01_INCLUDE_DIR NAMES gofs.h fdist.h unif01.h PATH_SUFFIXES testu01)
find_library(TestU01_probdist_LIBRARY NAMES probdist testu01probdist)
find_library(TestU01_mylib_LIBRARY NAMES mylib testu01mylib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TestU01 REQUIRED_VARS TestU01_probdist_LIBRARY TestU01_mylib_LIBRARY
                                                        TestU01_INCLUDE_DIR)
mark_as_advanced(TestU01_INCLUDE_DIR TestU01_probdist_LIBRARY TestU01_mylib_LIBRARY)

if(TestU01_FOUND)
  if(NOT TARGET TestU01::mylib)
    add_library(TestU01::mylib UNKNOWN IMPORTED)
    set_target_properties(TestU01::mylib PROPERTIES IMPORTED_LOCATION ${TestU01_mylib_LIBRARY}
                                                    INTERFACE_INCLUDE_DIRECTORIES ${TestU01_INCLUDE_DIR})
  endif()
  if(NOT TARGET TestU01::probdist)
    add_library(TestU01::probdist UNKNOWN IMPORTED)
    set_target_properties(TestU01::probdist PROPERTIES IMPORTED_LOCATION ${TestU01_probdist_LIBRARY}
                                                       INTERFACE_INCLUDE_DIRECTORIES ${TestU01_INCLUDE_DIR}
                                                       INTERFACE_LINK_LIBRARIES "TestU01::mylib;m")
  endif()
endif()
