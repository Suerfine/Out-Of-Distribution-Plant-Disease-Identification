## Overview
This document provides an overview of the Vision-Language Model (VLM) implemented for the Out-Of-Distribution Plant Disease Identification project.
    
- What is VLM? and what does it offer to improve on the baseline approach?
    - Vision Language Models (VLMs) consist of a vision and language encoder. It maps images and text into a shared embedding space. This allows for the model to learn a joint representation of both modalities, which means that the model can understand the relationship between images and their corresponding textual descriptions. This is paricularly useful for zero-shot learning, where the model can classify unseen classes based on their textual descriptions.
    - The baseline approach struggle with unseen diseases classification, this is where VLMs can help by leveraging textual descriptions of diseases to improve classification performance on unseen classes.

- Improving the VLM
  - There's issue with the current base VLM model, which is that it is not fine-tuned on the specific plant disease dataset. I approach the improvement of this model is by focusing on two main components of the VLM: the image encoder and the text encoder. 
    - Image Embedding
    - Text Embedding
      - Knowledge Base
      - Prompt Learning