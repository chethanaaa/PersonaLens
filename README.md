# PersonaLens: Behavioral Analysis for Interviews

## **Overview**
PersonaLens provides a comprehensive framework for analyzing interview performance by leveraging Generative AI and multimodal data (audio, video, and text). This system integrates insights from large language models (LLMs) to generate detailed, objective, and actionable reports based on candidate behavior.

Link to visit the POC: https://personalens.streamlit.app

For Access to the data please refer to the GDrive Data.txt for the google drive link.
---

## **Abstract**
Interviews play a critical role in hiring but often rely on subjective verbal responses, overlooking non-verbal cues like tone, body language, and emotional expressions. PersonaLens addresses these limitations by analyzing multimodal data from post-interview recordings (sourced from YouTube) using cutting-edge LLMs. By parsing insights from audio, video, and text into an LLM, the system generates a cohesive final report summarizing candidate performance. This ensures unbiased and data-driven decision-making, enabling recruiters to make informed choices.

---

## **Introduction**
### **Motivation**
- Interviews often overlook nuanced behavioral signals like vocal tone and body language.
- Subjective evaluations can lead to missed opportunities to assess candidates comprehensively.
- Non-verbal cues provide insights into engagement, confidence, and stress—elements critical for evaluating candidates.

### **Significance**
- PersonaLens enhances traditional interview practices by employing multimodal data and LLMs for behavioral analysis.
- This innovation bridges the gap between human judgment and data-driven insights, offering a more objective and fair evaluation system.

### **System Overview**
- **Audio Analysis**: Evaluates confidence, nervousness, and emotional states.
- **Video Analysis**: Detects visual cues such as blinks, gaze shifts, and yawns to measure engagement and focus.
- **Text Analysis**: Assesses clarity, sentiment, contradictions, and vocabulary in verbal responses.
- Insights are consolidated into a final LLM-generated report.

---

## **Literature Review**
### 1. **Leveraging Multimodal Behavioral Analytics for Automated Job Interview Performance Assessment and Feedback**
- **Authors**: Anumeha Agrawal et al.
- Multimodal framework evaluates engagement, eye contact, and speaking rate by analyzing video, audio, and text data.
- [Link](https://arxiv.org/abs/2006.07909)

### 2. **Automated Analysis and Prediction of Job Interview Performance**
- **Authors**: Iftekhar Naim et al.
- Framework quantifies verbal and non-verbal behaviors to predict traits like friendliness and excitement.
- [Link](https://arxiv.org/abs/1504.03425)

### 3. **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**
- **Authors**: Hang Zhang, Xin Li, Lidong Bing
- Introduces Video-LLaMA for video understanding tasks, integrating audio-visual data with text instructions.
- [Link](https://arxiv.org/abs/2306.08243)

### 4. **HawkEye: Training Video-Text LLMs for Grounding Text in Videos**
- **Authors**: Yueqian Wang et al.
- Training paradigm for video-text LLMs enables precise grounding of text in video segments.
- [Link](https://arxiv.org/abs/2306.16478)

---

## **Methods**
### 1. **Audio Analysis**
- **Features Extracted**: MFCC, spectral centroid, RMS energy, and tempo.
- **Behavioral Mapping**:
  - Higher spectral centroid: Nervousness.
  - RMS variations: Confidence levels.
- **LLM Integration**: Features parsed into Perplexity’s API for summaries.

### 2. **Video Analysis**
- **Event Detection**:
  - Blinks: Engagement.
  - Gaze shifts: Focus.
  - Yawns: Fatigue.
- **LLM Integration**: Logs parsed into Perplexity for behavioral insights.

### 3. **Text Analysis**
- **Focus Areas**:
  - Clarity, sentiment, contradictions, and vocabulary richness.
- **LLM Integration**: Perplexity (`llama-3.1-sonar-small-128k-online`) generated insights.

### 4. **Final Summarization**
- Multimodal insights parsed into an LLM for cohesive reporting.

### **Dataset**
- Interviews sourced from YouTube provided diverse candidate scenarios.

---

## **Results**
### **Audio Analysis**
- Spectral centroid variations correlated with nervousness.
- RMS energy indicated confidence improvements over time.

### **Video Analysis**
- High engagement in early stages, with fatigue observed later.
- Blinks: 18 | Gaze Outside Frame: 6 | Yawns: 4.

### **Text Analysis**
- Clarity rated high overall; minor contradictions identified.
- Sentiment revealed confidence and positivity in tone.

### **Final Summary Example**:
*"The candidate demonstrated strong engagement and confidence, with initial nervousness diminishing over time. Non-verbal cues suggested focus and attentiveness, while verbal responses were clear but lacked structural coherence in some instances."*

---

## **Conclusion**
### **Implications**
- Integrating multimodal insights ensures objective evaluations.
- LLMs like Perplexity unify data into actionable summaries.

### **Future Work**
- Incorporate advanced behavioral metrics like speech tempo and micro-expressions.
- Extend support to real-time live interview analysis.

### **Broader Impact**
- Revolutionizes recruitment by eliminating biases.
- Potential applications in therapy, education, and customer service for behavioral assessments.

---

## **Contact Information**
- **Chethana Saligram**
- **Shravan Srinivasan**  
