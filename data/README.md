## MMLU Data

This datasets is broken into two sections: training (unlabeled subjects) and validation/testing (labeled subjects)

### 1) Training
There are `99,842` training multiple choice questions, each with a `question`, a list of possible answers (called `choices`), and an `answer`.
<br/>
These questions do **NOT** have a labeled category. It says that they came from ARC, MC_TEST, OBQA, RACE, etc. (which must each be multiple choice test or banks of questions).
> I went through and used the Facebook classifier to give each quesiton a subject according to the below subjects, but **I added these**, they were not native to the data.

<br/>

### 2) Subject-specific Questions
There are also subject-specific questions, which are saved in the validation and testing datasets.
These are split accross `57` subjects, which are:


```
['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
```

Each subject has a set of `validation` and a set of `test` questions. 
<br/>
`Validation:` There are `14,042` validation questions total.
Each subject has *at least* `100` of them, with an average of `246` for each subject.
<br/>
`Test:` There are `1,531` test questions total.
Each subject has *at least* `8` of them, with an average of `27` for each subject.
