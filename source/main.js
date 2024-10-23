/*
Copyright (C) 2024 Gregory Teicher

Author: Gregory Teicher

This file is part of Marigold.

Marigold is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Marigold is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Marigold.
If not, see <https://www.gnu.org/licenses/>.
*/

import { BrowserCompatibility } from "./browser-compatibility.js";
import { LabelingSection } from "./labeling/labeling-section.js";
import { TrainingSection } from "./training/training-section.js";
import { AnalyzingSection } from "./analyzing/analyzing-section.js";


function optInToManualScrollRestoration() {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }
}


function initializeScrollToOnboardingButton() {
  if ("scroll" in window && "scrollX" in window && "scrollY" in window) {
    const scrollToOnboardingButton = document.querySelector("#home-scroll-to-onboarding-button");

    scrollToOnboardingButton.addEventListener(
      "click",
      (event) => {
        const onboarding = document.querySelector("#home-onboarding");
        window.scroll(
          {
            top: window.scrollY + onboarding.getBoundingClientRect().top + 1,
            left: window.scrollX,
            behavior: "smooth"
          }
        );
      }
    );

    scrollToOnboardingButton.removeAttribute("disabled");
  }
}


document.addEventListener(
  "DOMContentLoaded",
  async (event) => {
    optInToManualScrollRestoration();
    initializeScrollToOnboardingButton();

    {
      const response = await fetch("./labeling/labeling.html");
      const text = await response.text();
      document.querySelector("#labeling").innerHTML = text.trimEnd();
    }
    {
      const response = await fetch("./training/training.html");
      const text = await response.text();
      document.querySelector("#training").innerHTML = text.trimEnd();
    }
    {
      const response = await fetch("./analyzing/analyzing.html");
      const text = await response.text();
      document.querySelector("#analyzing").innerHTML = text.trimEnd();
    }

    const newDatasetButton = document.querySelector("#home-start-new-dataset-button");
    const openDatasetButton = document.querySelector("#home-open-existing-dataset-button");
    const newModelButton = document.querySelector("#home-start-new-model-button");
    const openModelButton = document.querySelector("#home-open-existing-model-button");
    const newAnalysisButton = document.querySelector("#home-start-new-analysis-button");
    const openAnalysisButton = document.querySelector("#home-open-existing-analysis-button");

    const browserCompatibility = new BrowserCompatibility();
    if (!browserCompatibility.seemsOkay()) {
      const buttons = [
        newDatasetButton,
        openDatasetButton,
        newModelButton,
        openModelButton,
        newAnalysisButton,
        openAnalysisButton
      ];
      for (const button of buttons) {
        button.addEventListener(
          "click",
          (event) => {
            browserCompatibility.showDialog();
          }
        );
      }
    }
    else {
      const block = () => {
        document.querySelector("#generic-dialog-close-button").addEventListener(
          "click",
          (event) => { document.querySelector("#generic-dialog").close(); },
          { once: true }
        );
        document.querySelector("#generic-dialog-heading").textContent = "Not enabled";
        document.querySelector("#generic-dialog-blurb").textContent = "Sorry, that feature is still under construction!";
        document.querySelector("#generic-dialog").showModal();
      };

      const labelingSection = new LabelingSection();
      newDatasetButton.addEventListener(
        "click",
        (event) => { labelingSection.maybeStartNewFile(); }
      );
      openDatasetButton.addEventListener(
        "click",
        (event) => { labelingSection.maybeOpenExistingFile(); }
      );

      const trainingSection = new TrainingSection();
      newModelButton.addEventListener(
        "click",
        (event) => { trainingSection.maybeStartNewFile(); }
      );
      openModelButton.addEventListener(
        "click",
        (event) => { block(); }
      );

      const analyzingSection = new AnalyzingSection();
      newAnalysisButton.addEventListener(
        "click",
        (event) => { analyzingSection.maybeStartNewFile(); }
      );
      openAnalysisButton.addEventListener(
        "click",
        (event) => { block(); }
      );
    }

    newDatasetButton.removeAttribute("disabled");
    openDatasetButton.removeAttribute("disabled");
    newModelButton.removeAttribute("disabled");
    openModelButton.removeAttribute("disabled");
    newAnalysisButton.removeAttribute("disabled");
    openAnalysisButton.removeAttribute("disabled");
  }
);
