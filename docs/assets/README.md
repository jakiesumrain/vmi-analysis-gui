# Documentation assets

- [Workbench artwork](vmi-workbench-art.png) is a decorative raster illustration made with built-in image generation. The image depicts a stylized instrument without measured data.
- [Agent approval](agent-approval.jpg) and [Ion TOF workbench](ion-tof-workbench.jpg) are browser captures from the running app. They use a simulated model endpoint and a synthetic LUNA-format HDF5 fixture with two peak populations. The example uses raw pixel hits, TDC1 rising, 10 ns bins, and a 0–12,000 ns TOF interval. Display paths were shortened and the local console command was omitted.
- Edit the [Graphviz sources](../diagrams) and regenerate the SVG with `dot -Tsvg` when the design changes.

## Artwork prompt

> Create a wide 3:1 editorial scientific illustration for VMI Analysis, an electron and ion velocity-map imaging workbench. Dark ink navy background, crisp luminous concentric velocity-map detector rings at the center-left in restrained warm amber and cobalt, a second sparse set of ion arrival traces tapering to the right, a few precise pale-blue coordinate grid marks. Elegant minimal scientific instrument aesthetic, wide breathing room, restrained glow, no text, no numbers, no logos, no people, no app UI, no fabricated data captions. This is decorative art and should read well scaled down to a GitHub README header.
