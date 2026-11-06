"""Compatibility launcher; implementation is in src/vmi_analysis."""
import runpy

if __name__ == "__main__":
    runpy.run_module("vmi_analysis.imaging.h5_accumulate_fast_gmd", run_name="__main__", alter_sys=True)
else:
    globals().update(runpy.run_module("vmi_analysis.imaging.h5_accumulate_fast_gmd", run_name=__name__))
