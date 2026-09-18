# Publication Readiness

| Item | Current Status | Required For Paper Code Release? | Required For Frozen Reproduction? | Action Needed | Owner |
| --- | --- | --- | --- | --- | --- |
| Source code | Public package and scripts are self-contained under `src/viunmark/` and `scripts/`. | Yes | Yes | Review final API and package metadata. | Author |
| README | Updated to distinguish manifest inspection, real asset verification, unavailable retraining, diagnostics, and sealed scoring. | Yes | Yes | Final citation details after paper metadata is fixed. | Author |
| Notebook | One orchestrator notebook calls public scripts and uses configurable roots. | Yes | Yes | Optional execution after assets are supplied. | Author |
| Configs | UIT-VSFC public config is present. | Yes | Yes | Confirm public path defaults are acceptable. | Author |
| Unit tests | Public tests cover method invariants and hardened reproduction contracts. | Yes | No | Run in the final release environment. | Author |
| Stage-I checkpoints | Not shipped; two SHA-256 identities and expected paths are recorded. | No | Yes | Supply `ViUnMark-Gate` and `ViUnMark-Scale` checkpoints under the configured asset root. | Author |
| Stage-II heads | Not shipped; twenty SHA-256 identities, seeds, and selected boundaries are recorded. | No | Yes | Supply five heads for each branch under the configured asset root. | Author |
| Dataset | Raw UIT-VSFC data is not redistributed. | No | Yes | User supplies dataset under its own terms and configured schema. | User |
| Official result records | Canonical numeric files are not present in current Git; known SHA-256 identities are recorded. | Yes | Yes | Supply author-approved result artifacts before rendering paper tables. | Author |
| Diagnostics | Public commands are available; several analyses require external frozen artifacts for full numerical reproduction. | Yes | Some | Keep the diagnostic executability matrix in the release review notes current. | Author |
| Software license | `PUBLIC_CODE_LICENSE=UNRESOLVED`; no repository license was found. | Yes | Yes | Choose and add a project software license before publication. | Author |
| Citation metadata | Placeholder only. | Yes | No | Add final bibliographic metadata. | Author |
| PhoBERT attribution | Model identity is recorded; redistribution terms are not altered here. | Yes | Yes | Confirm license/reference text against the upstream model card. | Author |
| Checkpoint redistribution | External assets are not shipped in this export. | No | Yes | Decide release channel and publish checksums without inventing URLs. | Author |
