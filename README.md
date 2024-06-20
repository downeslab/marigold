# Marigold: Zebrafish pose tracking made easier

Marigold is a free and open source web app for analyzing zebrafish behavior. It is being developed by Gregory Teicher (gteicher at umass dot edu) in the [Downes Laboratory](https://www.downeslab.org/) at University of Massachusetts Amherst.

Marigold is compatible with a wide range of experimental paradigms, particularly at embryonic and larval stages. As long as each region of interest only contains a single fish, it is likely that Marigold can track it.

Check out our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.05.31.596910v1) for more info! (And please remember to cite the preprint if you wind up using Marigold in your research!)

Marigold is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html) or later. You are currently viewing its source code on GitHub, where we also host the [web app](https://downeslab.github.io/marigold/) itself.

Currently, Marigold should be regarded as **alpha-level software.** That means it is still missing elements of the planned functionality, contains some known bugs, and is likely to also contain unknown bugs. Additionally, it may not be compatible with future versions of itself. Nevertheless, since we've found Marigold useful even in its current state and hope you will too, we've released it as-is while we track down bugs and implement the remaining functionality.

Here's a list of some current issues and limitations:

- Compatible only with movie files encoded using H.264/AVC compression in an MP4 container. We hope to add support for additional movie formats soon. In the meantime, if you run into issues with your movie files we recommend trying to convert them to the recommended format. To help with troubleshooting this issue and to rule out other problems with the web app, we've provided examples of movies that are known to work well in the `examples` folder of this repository. Feel free to reach out to the developer (email above) if you're not sure how to convert your movies to the recommended format.

- During tracking of movies with a large number of frames, the speed of analysis slows down as the analysis progresses. This effect shouldn't be particularly noticeable for movies that are only a few thousand frames in length, but may become problematic for movies containing considerably more frames. We have some leads on this issue and hope to find time to fix it soon. In the meantime, consider using the frame selection options to analyze extremely long movies in "chunks" of a couple thousand frames at a time.
