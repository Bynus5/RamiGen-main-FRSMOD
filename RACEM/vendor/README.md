# RACEM runtime vendor sources

Only runtime sources and upstream licenses are kept here; original standalone
training, evaluation, datasets, demos and deployment assets were archived outside
the project. Model mathematics are unchanged. Checkpoints live in `ckpts/racem/`
at the project root, not in this directory.

- `sauge/`: legacy SAUGE sources retained for reference, not used by RACEM.
  Active edge extraction uses the supplied Paddle HED in `RACEM/edge_detector/`.
- `sam2/`: SAM2 image predictor/model code and SAM2.1 Hiera-large config.
  Source: https://github.com/facebookresearch/sam2.
- `midas/`: MiDaS model loader, DPT BEiT-large-512 and modules imported by it.
  Source: https://github.com/isl-org/MiDaS.

`sam2` and `midas` keep their upstream import names; RACEM
loads these bundled sources lazily. Do not substitute an unrelated installed
package with the same name. Original licenses are retained in each directory.
