<!-- BEGIN climate_risk -->

## climate_risk

Every loader takes a `cache_dir` argument. There is no default, no environment variable and no
project-root search, so resolve the path once and pass it down.

EM-DAT, GADM, Geo-Disasters and the Penn World Table cannot be downloaded by code. A loader that
needs one raises with the license and the exact path the file belongs at. Do not write a downloader
for them.

This package ships agent skills covering the above in more detail. `install-climate-risk-skills`
places them in `.claude/skills/`, each with the user-guide pages it refers to.

<!-- END climate_risk -->
