#------------------------------------------------------------------------------
# Add a library.
#
# This convenience function defines a library to be built, generates the export
# header for the library, sets up the install rules for the library and its
# installed headers, and sets up exported targets for the library.
#
# Arguments:
#   <NAME> - Name of library to build.
#
#   [SOURCES] <sources...>
#     List of library source files, including private headers.
#
#   INSTALL_HEADERS <headers...>
#     List of public headers that will be installed along with the library.
#------------------------------------------------------------------------------
function(drake_add_library NAME)
  cmake_parse_arguments("" "" "" "SOURCES;INSTALL_HEADERS" ${ARGN})

  add_library(${NAME} ${_UNPARSED_ARGUMENTS} ${_SOURCES} ${_INSTALL_HEADERS})
  target_include_directories(${NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR})

  generate_export_header(${NAME})
  list(APPEND _INSTALL_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_export.h)
  set_target_properties(${NAME} PROPERTIES PUBLIC_HEADER "${_INSTALL_HEADERS}")

  install(TARGETS ${NAME}
    EXPORT drakeTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    ARCHIVE DESTINATION lib${LIB_SUFFIX}
    PUBLIC_HEADER DESTINATION include/drake
    INCLUDES DESTINATION include)
endfunction()
