SYSTEM = """Internet Text Register Classification Expert

You are a specialized text classifier that ALWAYS provides responses in this exact format:
DIGIT: BRIEF_REASON

Rules:
1. DIGIT must be exactly one of the allowed numbers for each question
2. A colon must separate the digit from the reason
3. BRIEF_REASON must be 3-10 words explaining the key determining feature
4. No other text or explanations are allowed

For each text analysis:
- Read the entire text carefully
- Focus on dominant characteristics
- Ignore supplementary elements (comments, ads)
- Choose the MOST prominent category
- Always output using the exact format: DIGIT: BRIEF_REASON

Examples of correct responses:
1: [Brief explanation]
2: [Brief explanation]
0: [Brief explanation]

NEVER provide additional explanation or text beyond this format."""

INITIAL_PROMPT = """Here is the document to classify:

```
{document}
```

First, read the entire document carefully. Then determine:
Does this document contain enough substantive text to make a reliable register classification?
- If yes, choose 1
- If no, choose 0.

Output your answer in this format:
DIGIT: BRIEF_REASON"""

MODE_OF_PRODUCTION = """Analyze the text's primary mode:

(1) SPOKEN ORIGIN: 
   - A text that is composed of more than 50% spoken material
   - Interview, typically dialogic in a question-answer format
   - Formal speeches, such as ones held by politicians
   - TV/movie transcripts or Youtube video transcripts
   - Other clearly spoken origin texts
   
(2) WRITTEN ORIGIN:
   - Text originated in written form
   
(3) MACHINE/TEMPLATE:
   - Formulaic/repetitive patterns
   - Unnatural language flow
   - Template-based structure

Output your answer in this format:
DIGIT: BRIEF_REASON"""

IS_INTERACTIVE = """Determine if this is primarily an interactive or non-interactive text:

(1) INTERACTIVE DISCUSSION:
    - Interactive forum discussions with discussion participants and possibly other readers
    - Question-answer forums, where one person asks a question and one or several answer it
    - The text only consists of comments that belong/are related to a blog post or an article, but the blog post or article is missing
    - Text consisting of a single comment (when it’s clear that it is a comment)
    
(2) NON-INTERACTIVE TEXT:
    - Single-direction communication (single author or coauthors; e.g. blog post, article)
    - Reader comments (if present) are separate from main text

Note: A non-interactive text that has comments section should still be classified as (2).

Output your answer in this format:
DIGIT: BRIEF_REASON"""

IS_INTERVIEW = """Determine if this text is primarily an interview:

(1) INTERVIEW:
    - Typically one interviewer and one interviewee
    - E.g. participants a radio show host / journalist and a famous person or an invited expert
    - Most interviews are dialogic and have a question-answer format

(0) NOT INTERVIEW:
    - Other types of spoken texts that are not interviews

Note: Text with occasional quoted responses is not an interview!

Output your answer in this format:
DIGIT: BRIEF_REASON"""

COMMUNICATIVE_PURPOSE = """Determine the text's DOMINANT purpose:

(1) NARRATE: Purpose to narrate or report on events
   - E.g. news articles, blogs, sports reports
   
(2) INFORM: Present facts/information
   - Objective presentation
   - Focus on accuracy
   - E.g. descriptions of a thing or person, research articles, encyclopedias
   
(3) EXPRESS OPINION: Share views
   - Personal perspective
   - Evaluative language
   - E.g. reviews, opinion blogs, religious sermons
   
(4) PERSUADE: purpose to describe or explain facts with intent to persuade or market
   - Arguments for position
   - Texts that sell or promote a service, product or upcoming event
   - E.g. sales descriptions, editorials
   
(5) INSTRUCT: Purpose to explain how-to or instructions
   - Step-by-step guidance
   - Clear directions
   - Objective instructions on how to do something.
   
(6) LYRICAL: Artistic expression
   - Songs or poems or other texts that are written in a lyrical way
   - Typically, written by professional songwriters and poets, but they are posted online by fans and online contributors

Output your answer in this format:
DIGIT: BRIEF_REASON"""


PURPOSE_SPECIFIC_PROMPTS = {
    "1": """For NARRATIVE texts, identify the dominant type:

(1) NEWS REPORT/BLOG:
   - News reports written by journalists and published by news outlets
   - Releases and newsletters published by sports associations, companies, etc.
   - Weather forecasts
   - Text purpose is to report on recent events
   - Typically professionally written and time-sensitive - published and read as fast as possible

(2) SPORTS REPORT:
   - Text purpose is to report on a recent sports event
   - Typically written by professional journalists, but can also be published by amateur writers, for instance on sports club home pages
   - Time-sensitive – published and read as fast as possible
   - Note that not all texts on the topic of sport are automatically annotated as Sports report. If the purpose of the text is not to report on a sports event, other register classes should be considered. For example, an article about politics in sports could be annotated as (1) News report.

(3) NARRATIVE BLOG: 
   - Personal blogs, travel blogs, lifestyle blogs, blogs written by communities
   - Purpose to narrate / comment about events experienced by the writer(s)
   - Typically amateur writers
   - Can include interactive aspects, such as comments following the blog post

(4) OTHER NARRATIVE:
   - Narrative texts that are not News reports, Sports reports or Narrative blogs
   - Text purpose is to narrate or report on an event
   - Focus on objective, factual, neutral content
   - Texts such as short stories, fiction, magazine articles, other online articles

Output your answer in this format:
DIGIT: BRIEF_REASON""",
    "2": """For INFORMATIONAL texts, identify the dominant type:

(1) ENCYCLOPEDIA:
   - Texts that describe or explain a topic
   - Objective is to synthesize the current state of knowledge from all available studies
   - Typically written by a collaborative group of co-authors
   - A “wiki” platform; either Wikipedia or a similar one
   - A dictionary entry
   - Many of the articles are biographical describing a person’s life and accomplishments

(2) RESEARCH ARTICLE:
   - Describes a research study, including the motivation for the study, the methods used, and the major research findings
   - Written either by an individual or a collaborative group of co-authors, associated with an academic institution
   - Target audience specialists
   - Also dissertations and theses are included in this group

(3) DESCRIPTION OF A THING OR PERSON
   - Texts describing a thing or a person (excluding Encyclopedia articles)
   - A variety of documents ranging from administrative websites describing taxation to health care officials describing illnessess and associations describing their activities
   - This category includes also job descriptions and notices of open tender

(4) FAQ:
   - Documents sructured as questions-and-answers
   - Text purpose to provide specific information about something
   - Websites with procedural information often have special pages with FAQs, anticipating questions that end-users may have
   - The author is usually associated with an institutional or commercial site

(5) LEGAL TERMS:
   - Any document describing legislation
   - Texts belonging to Legal terms and conditions are official by nature
   - E.g., privacy policies, long cookie descriptions, texts describing terms and conditions, bills, rules of an association
   - For rules of a game, choose (6) Other Informational

(6) OTHER INFORMATIONAL:
   - Texts that describe or explain something but are none of the above
   - For instance, course materials, test papers, meeting minutes, and descriptive reports
   - Also informational blogs informing the reader
   - Presented as objective information rather than personal opinion

Output your answer in this format:
DIGIT: BRIEF_REASON""",
    "3": """For OPINION texts, identify the dominant type:

(1) REVIEW:
   - Texts evaluating the quality of a product or a service
   - Can be written on a personal, institutional, or commercial website

(2) OPINION BLOG:
   - Blogs written to express the writer’s / writers’ opinion
   - Typically written by an amateur writer, such as a politician
   - Typical topics include politics, governmental policies and social issues
   - The author does not need to have any special expertise or credentials
   - Focus on present time orientation
   - Expressions of evaluation and stance, overt argumentation

(3) RELIGIOUS:
   - Denominational religious blog, sermon or basically any other denominational religious text
   - Focus on denominational: texts describing a religion is not in this category

(4) ADVICE:
   - Based on a personal opinion
   - Purpose to offer advice that leads to suggested actions and solves a particular problem
   - Objective instructions aren not in this category
   - Often associated with an institutional or commercial site
   - Directive and suggesting actions for the reader
   - Typical topics include healthcare, finding a job, parenting, training for a sport

(5) OTHER OPINION:
   - Text expressing the writer’s or writers’ opinion that are none of the above
   - For example an opinion piece

Output your answer in this format:
DIGIT: BRIEF_REASON""",
    "4": """For PERSUASIVE texts, identify the dominant type:

(1) SALES DESCRIPTION:
   - Texts describing something with the purpose of selling
   - Overtly marketing, but money does not need to be mentioned
   - E.g., book blurbs (including library recommendations), product descriptions, marketing a service

(2) EDITORIAL:
   - Purpose to persuade the reader by using information and facts
   - Typically written by a professional on a news-related topic
   - Can be associated with a newspaper or magazine

(3) OTHER PERSUASIVE:
   - Texts with intent to persuade that are none of the above
   - Unlike Sales description, these texts are not overtly marketing a product, service or similar
   - For instance, persuasive and argumentative essays, texts on public health promoting healthy lifestyles, texts that advertise an upcoming event, market an enterprise without overtly selling (e.g. description of an enterprise) or product placement, etc.

Output your answer in this format:
DIGIT: BRIEF_REASON""",
    "5": """For INSTRUCTIONAL texts, identify the dominant type:

(1) RECIPE:
   - Step-by-step instructions on how to prepare or cook something, typically food
   - Should include at least the ingredients and/or the actual instructions

(2) OTHER INSTRUCTIONS:
   - How-to instructions that are not Recipes
   - Objective instructions on how to perform a task, often step-by-step
   - E.g., rules of a game, tutorials, instructions on how to fill a form
   - Can be written on a personal, commercial or institutional website

Output your answer in this format:
DIGIT: BRIEF_REASON""",
}
