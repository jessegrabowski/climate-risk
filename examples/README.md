# Gallery examples

Notebooks here are staged into the documentation gallery at build time. A notebook's path decides
where it lands: `examples/<section>/name.ipynb` puts it under a section heading named after the
folder, and `examples/name.ipynb` puts it on the page with no heading.

Ship them pre-executed with their outputs committed. Nothing runs them at build time, and the last
image output of a notebook becomes its gallery thumbnail.

This tree is separate from `notebooks/`, which holds research working files. These are a curated
artifact and are held to the documentation's standards.
