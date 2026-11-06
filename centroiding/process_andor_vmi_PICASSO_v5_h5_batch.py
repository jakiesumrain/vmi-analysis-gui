"""Compatibility launcher; implementation is in src/vmi_analysis."""
import runpy

if __name__ == "__main__":
    runpy.run_module("vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch", run_name="__main__", alter_sys=True)
else:
    globals().update(runpy.run_module("vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch", run_name=__name__))
