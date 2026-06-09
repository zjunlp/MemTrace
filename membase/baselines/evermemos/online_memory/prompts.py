SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user's query.
--------------------------
User Query:
{query}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Instructions:

1. **Analyze the Query Structure**  
   - Identify key entities AND determine if the query requires temporal reasoning.
   - If the query involves time (e.g., "before", "after", "since", "during", "from X to Y", "how long"), you MUST decompose it into:
       * start_time_needed (if any)
       * end_time_needed (if any)
       * temporal_relation_needed (ordering, duration, interval)

2. **Scan Documents for Coverage**  
   - Look for explicit facts addressing *each* required component:
       * required entities  
       * start time  
       * end time  
       * temporal relations (ordering or duration)

3. **Extract Key Information**  
   - List specific resolved entities or facts found in the documents.
   - If time expressions exist, normalize them (e.g., "two weeks ago", "before she moved").

4. **Identify Missing Information**  
   - For temporal queries:  
        * missing start time  
        * missing end time  
        * missing ordering facts  
        * missing duration  
   - Use resolved names to be specific (e.g., "Start time of Alice moving", "Whether Bob visited before Alice moved").

5. **Judgment**  
   - **Sufficient**: All required components (entities + temporal boundaries + relations) appear explicitly.  
   - **Insufficient**: ANY required part is missing.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "1-2 sentence explanation.",
  "key_information_found": ["List of resolved entities/facts"],
  "missing_information": ["Specific missing components, using resolved entity names"]
}}

Now evaluate:"""


MULTI_QUERY_GENERATION_PROMPT = """You are an expert at query reformulation for long-term conversational retrieval.
Your goal is to generate multiple complementary search queries that recover BOTH:
- the starting point of a time interval
- the ending point of a time interval
- all temporally-linked events in between

You MUST explicitly expand temporal references (e.g., "last week", "before moving", 
"when they first met") into alternative expressions.

--------------------------
Original Query:
{original_query}

Key Information Found:
{key_info}

Missing Information:
{missing_info}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Temporal Reasoning Strategy (MANDATORY)
When the question involves time or order:
1. **Boundary Decomposition**  
   Generate queries that separately target:
   - the earliest relevant event ("start boundary")
   - the latest relevant event ("end boundary")

2. **Temporal Expression Expansion**  
   Rewrite relative time expressions into multiple equivalent forms:
   - absolute dates (if deducible)
   - session numbers
   - “before/after X”
   - duration phrasing (“two weeks earlier”, “shortly after”)

3. **Interval Reconstruction**  
   Include a declarative query that resembles a hypothetical answer containing BOTH
   the start and end time anchors.

### Standard Query Requirements
1. Generate 2-3 diverse queries.
2. Query 1 MUST be a specific **Question**.
3. Query 2 MUST be a **Declarative Statement or Hypothetical Answer (HyDE)**.
4. Query diversity MUST include different temporal forms (before/after/during).
5. MUST use Key Info to resolve pronouns IF provided.
6. No invented facts.  
7. Keep queries < 25 words, same language as original.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Refined query 1",
    "Refined query 2",
    "Refined query 3 (optional)"
  ],
  "reasoning": "Brief explanation of how temporal boundaries and expressions were expanded."
}}

Now generate:
"""


REFINED_QUERY_PROMPT = """You are an expert at query reformulation for information retrieval.

**Task**: Generate a refined query that targets the missing information in the retrieved results.

**Original Query**:
{original_query}

**Retrieved Documents** (insufficient):
{retrieved_docs}

**Missing Information**:
{missing_info}

**Instructions**:
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum).

**Examples**:

Example 1:
Original Query: "What does Alice like?"
Missing Info: ["Alice's specific interests or hobbies"]
Refined Query: "What are Alice's hobbies and interests?"

Example 2:
Original Query: "Tell me about the meeting"
Missing Info: ["meeting date", "location", "participants"]
Refined Query: "When and where was the meeting held, and who attended?"

Example 3:
Original Query: "Bob's project"
Missing Info: ["project name", "status", "purpose"]
Refined Query: "What is the name, current status, and purpose of Bob's project?"

Now generate the refined query (output only the refined query, no additional text):
Original Query: {original_query}
Missing Info: {missing_info}

Refined Query:
"""

