ExternalProject_Add(openblas
    URL ${openblas_url}
    URL_MD5 ${openblas_md5}
    DOWNLOAD_NAME ${openblas_dlname}
    ${COMMON_EP_ARGS}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make NO_SHARED=1
    INSTALL_COMMAND make NO_SHARED=1 PREFIX=${CMAKE_INSTALL_PREFIX} install
)

extract_license(openblas
    LICENSE
)
