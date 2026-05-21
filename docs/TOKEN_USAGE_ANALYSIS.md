# Token Usage Analysis
- This analysis covers all llm generation that used in pipelines
- This token usage are collected using internal TrackToken class
- Cost calculation is based on OpenAI API pricing (gpt-4o-mini) and converted into IDR (Rp17200 / usd)


## Result
Result 1:
- cv_length = 2328
- jr_length = 269
- jr_n = 5

![v1.1_token_analysis_1](images/v1.1_token_analysis_1.png)

Result 2:
- cv_length = 4975
- jr_length = 690
- jr_n = 10

![v1.1_token_analysis_1](images/v1.1_token_analysis_2.png)

Result 3:
- cv_length = 848
- jr_length = 1198
- jr_n = 15

![v1.1_token_analysis_1](images/v1.1_token_analysis_3.png)

## Analysis 1 (Dominant Token Consumer Analysis)
Analysis:
- Input Token Dominance:
    - inf_evaluation is the dominant input token consumer across all analyses
    - inf_evaluation contributes approximately:
  - ~55% of total input tokens
  - ~45–62% of total output tokens

- Output Token Dominance:
    - inf_evaluation is also the dominant output token generator across all analyses
    - Output token generation is primarily driven by:
        - reasoning generation
        - evidence explanation
        - structured evaluation responses

## Analysis 2 (Token Scaling Analysis)
Analysis:
- pre_parse token usage scales mainly with CV length
- inf_chunk token usage scales with:
  - job requirement length
  - number of job requirements
- inf_evaluation token usage scales most heavily because it combines:
  - retrieved evidence
  - decomposition context
  - reasoning generation
  - repeated evaluation workflow
- inf_report token usage scales relatively moderately compared to evaluation stages

Conclusion:
- Job requirement complexity and quantity have stronger impact on total token usage than CV size alone
- inf_evaluation is the primary contributor to token scaling and inference cost growth

## Analysis 3 (Input Output Token Ratio)
Analysis:
- pre_parse (Raw CV → Structured CV):
    - Output/Input ratio:
        - ~0.45–0.66
    - pre_parse is a transformation-heavy stage that converts raw CV text into structured information
    - Output token generation scales with the amount of extracted structured information

- inf_chunk (Raw Job Requirement → Decomposed Requirements)
    - Output/Input ratio:
        - ~0.15–0.18
    - inf_chunk is highly input-heavy because of decomposition instructions and structured prompting overhead
    - Generated output remains relatively compact compared to total prompt size

- inf_evaluation (Requirement Evaluation)
    - Output/Input ratio:
        - ~0.23–0.28
    - inf_evaluation is both reasoning-heavy and input-heavy
    - Token usage scales with:
        - number of job requirements
        - retrieved evidence quantity
        - reasoning complexity

- inf_report (Evaluation Results → Recruiter-Style Report)
    - Output/Input ratio:
        - ~0.42–0.47
    - inf_report is generation-oriented and produces higher ratio because its task is to conclude input

## Analysis 4 (Separated Preprocess and Inference Pipeline)
### Configuration
- cv_length = 2328
- jr_length = 269
- jr_n = 5

### Result

![v1.2_token_analysis_4](images/v1.2_token_analysis_1.png)

### Analysis
- Average preprocess pipeline cost:
  - ~7.76 IDR
- Average inference pipeline cost:
  - ~27.78 IDR
- Average full pipeline cost:
  - ~35.53 IDR

### Cost Reduction Analysis
- Percentage contribution of preprocess pipeline:
  - (7.76 / 35.53) * 100% ≈ ~21.8%
- Percentage contribution of inference pipeline:
  - (27.78 / 35.53) * 100% ≈ ~78.2%

### Summary
- Separating the CV preprocessing stage from the inference pipeline avoids repeating expensive preprocessing operations for every evaluation
- Reusing stored CV artifacts reduces repeated inference cost by approximately ~22%
- This optimization is valuable because the same CV is often evaluated against multiple different job requirements