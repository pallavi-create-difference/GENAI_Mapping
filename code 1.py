import streamlit as st
import google.generativeai as genai
import os
import textwrap
import pandas as pd
import sys

# --- Configuration ---
# improve display
try:
    from IPython.display import display, Markdown
    ipython_display_available = True
except ImportError:
    ipython_display_available = False
    def display(obj): pass
    def Markdown(text): return text

# Model and File Paths
MODEL_NAME = 'gemini-1.5-flash-latest'
# --- Adjust these filenames if they differ ---
CLAIMS_DATA_CSV_PATH = 'CLAIM (1).csv' # Source for claim data & actual columns
POLICY_DATA_CSV_PATH = 'POLICY.csv' # Source for policy data & actual columns
PMT_SCHEMA_CSV_PATH = 'PMT_Schema3226573860721396623 1.csv' # Source for target schema column names
CLAIM_HTML_DICT_PATH = 'Claim.html' # Path to your HTML dictionary (OPTIONAL - used if parsing implemented)
# ---

SAMPLE_ROWS_TO_INCLUDE = 10

# --- !!! IMPORTANT: USE YOUR VALID API KEY !!! ---
API_KEY = "AIzaSyCytW7AGIOeqCt0b1F3_TL-yy6CnnL-dF0" # Replace with your key if needed
# ---

if not API_KEY or len(API_KEY) < 20 or API_KEY == "YOUR_API_KEY_HERE":
     print("ERROR: API_KEY variable is not set correctly in the script.")
     print("Please replace 'YOUR_API_KEY_HERE' with your actual, valid API key.")
     sys.exit(1)

# --- AI Model Setup ---
try:
    genai.configure(api_key=API_KEY)
    print("Google Generative AI configured successfully.")
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Generative model '{MODEL_NAME}' initialized.")
except Exception as e:
    print(f"ERROR: Failed to configure or initialize the AI model: {e}")
    print(f"Please check API key validity, network connection, and if model '{MODEL_NAME}' is available.")
    sys.exit(1)

# --- Data Reading Functions ---

def read_pmt_schema(filepath):
    """Reads the PMT Schema CSV and extracts TARGET column lists for Claim and Policy tables."""
    print(f"Reading TARGET schema definition from: {filepath}")
    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            df.columns = df.columns.str.strip()
        except UnicodeDecodeError:
            print(f"Warning: UTF-8 decoding failed for {filepath}. Trying latin-1.")
            df = pd.read_csv(filepath, encoding='latin-1')
            df.columns = df.columns.str.strip()

        required_table_col = 'TableName'
        required_field_col = 'Field Name' # Use the name with the space

        if required_table_col not in df.columns or required_field_col not in df.columns:
            print(f"ERROR: PMT Schema file '{filepath}' is missing required columns '{required_table_col}' or '{required_field_col}'.")
            print(f"Actual columns found: {df.columns.tolist()}")
            sys.exit(1)

        claim_target_cols = df[df[required_table_col] == 'Claim'][required_field_col].tolist()
        policy_target_cols = df[df[required_table_col] == 'Policy'][required_field_col].tolist()

        if not claim_target_cols:
            print(f"Warning: No TARGET columns found for TableName 'Claim' in {filepath}.")
        else:
            print(f"-> Found {len(claim_target_cols)} TARGET columns for 'Claim' table in schema.")

        if not policy_target_cols:
            print(f"Warning: No TARGET columns found for TableName 'Policy' in {filepath}.")
        else:
            print(f"-> Found {len(policy_target_cols)} TARGET columns for 'Policy' table in schema.")

        return claim_target_cols, policy_target_cols

    except FileNotFoundError:
        print(f"ERROR: PMT Schema file not found at '{filepath}'. Please check path.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"ERROR: PMT Schema file '{filepath}' is empty or invalid.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read PMT Schema file '{filepath}'. Error: {e}")
        sys.exit(1)


def read_csv_data_info(filepath, num_sample_rows=5):
    """Reads a data CSV file to get its actual columns and sample data rows."""
    print(f"Reading ACTUAL columns and sample data from: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding='utf-8', nrows=max(num_sample_rows, 50))
        actual_columns = df.columns.tolist()
        num_actual_samples = min(num_sample_rows, len(df))
    except UnicodeDecodeError:
        print(f"Warning: UTF-8 decoding failed for {filepath}. Trying latin-1.")
        df = pd.read_csv(filepath, encoding='latin-1', nrows=max(num_sample_rows, 50))
        actual_columns = df.columns.tolist()
        num_actual_samples = min(num_sample_rows, len(df))
    except FileNotFoundError:
        print(f"ERROR: Data CSV file not found at '{filepath}'. Please check path.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"ERROR: Data CSV file '{filepath}' is empty or invalid.")
        return [], "[File is empty]"
    except Exception as e:
        print(f"ERROR: Failed to read data CSV file '{filepath}'. Error: {e}")
        sys.exit(1)

    sample_data_df = df.head(num_actual_samples)
    sample_data_str = sample_data_df.to_string(index=False)
    print(f"-> Found {len(actual_columns)} ACTUAL columns. Using first {len(sample_data_df)} rows read as sample.")
    if len(df) == 0:
        sample_data_str = "[File is empty]"
        actual_columns = []
    return actual_columns, sample_data_str

# --- Representing the Target Claim Dictionary ---
# Ideally, load this from a pre-parsed file (e.g., CSV/JSON created from HTML)
# For demonstration, we create a multi-line string snippet.
# NOTE: This is just a SMALL sample. A more complete version would be better.
claim_dictionary_details_snippet = """
Field Name          | Inferred Data Type | Description
--------------------|--------------------|------------
AccidentType        | TypeKey            | Detailed accident type; augments LossCause.
AgencyId            | String             | An ID assigned to indicate company and office a claim is being submitted by, this data is used by ISO integration
AssignedByUser      | ForeignKey (User)  | User who assigned this entity.
AssignedGroup       | ForeignKey (Group) | Group to which this entity is assigned; null if none assigned
AssignedQueue       | ForeignKey (Queue) | Queue to which this entity is assigned...
AssignedUser        | ForeignKey (User)  | User to which this entity is assigned...
AssignmentDate      | DateTime           | Time when entity last assigned
AssignmentStatus    | TypeKey            | Typelist describing assignment status.
BenefitsStatusDcsn  | Boolean            | Indicates if the benefits decision has been made yet.
Catastrophe         | ForeignKey (Cat)   | Associated catastrophe.
ClaimantDenorm      | ForeignKey (Cont)  | Claimant FK denorm.
ClaimNumber         | String             | The external identifier of the claim. (Note: Data dictionary calls this 'claimnumber', target schema 'ClaimNumber')
ClaimSource         | TypeKey            | Information about how Claim was entered into the System.
ClaimTier           | TypeKey            | The tier of this claim, used to decide how to rate the claim metrics.
ClaimWorkComp       | ForeignKey (WC)    | Claim's worker's compensation data
CloseDate           | DateTime           | Date and time when this entity was closed.
ClosedOutcome       | TypeKey            | The outcome reached when closing the claim.
CoverageInQuestion  | Boolean            | Whether the claim is covered by the claimant's policies.
Currency            | TypeKey            | The currency for the claim, copied from the policy.
Description         | String             | Description of the accident or loss.
Fault               | Decimal/Percentage | Insured's probable percentage of fault.
FaultRating         | TypeKey            | Indicates in the insured is at fault.
LOBCode             | TypeKey            | Line of Business code; typically related to the policy.
LossCause           | TypeKey            | General cause of loss; dependent on loss type.
LossDate            | DateTime           | The date on which the loss occurred.
LossLocation        | ForeignKey (Addr)  | Location of the loss.
Policy              | ForeignKey (Policy)| The policy associated with this claim.
ReportedDate        | DateTime           | Date on which the loss was reported.
State               | TypeKey            | Internal state of the claim.
JurisdictionState   | TypeKey            | The state of jurisdiction. This indicates jurisdiction that covers the loss...
PMT_ID              | Integer            | (Assumed from schema) Unique identifier for the claim within the PMT system.
(Only includes a subset of fields for brevity in this example)...
"""
# --- Prompt Generation Functions ---

def create_data_dictionary_prompt(claim_target_cols, claim_sample,
                                  policy_target_cols, policy_sample, sample_rows):
    """Generates the data dictionary prompt, focusing on TARGET schema columns."""
    # This prompt remains focused on describing the TARGET schema from PMT,
    # using the sample data from CSVs for context. Minor refinement added.
    return textwrap.dedent(f"""
    Generate a data dictionary in Markdown table format describing the TARGET schema for the Claim and Policy tables, as defined in the `PMT_Schema*.csv` file.
    Use the definitive TARGET column names provided below.
    Leverage the sample data (extracted from `CLAIM (1).csv` and `POLICY.csv`) SOLELY as context to help infer likely data types (e.g., Integer, String, Date, Datetime, Float/Decimal, Boolean/Flag) and write meaningful descriptions for these TARGET columns based on their names and the observed sample values.

    **Table 1: Claim (Target Schema defined in PMT_Schema*.csv)**
    Definitive Target Columns: {', '.join(claim_target_cols) if claim_target_cols else 'None Provided'}
    Contextual Sample Data (first {sample_rows} rows from `CLAIM (1).csv`):
    ```
    {claim_sample}
    ```

    **Table 2: Policy (Target Schema defined in PMT_Schema*.csv)**
    Definitive Target Columns: {', '.join(policy_target_cols) if policy_target_cols else 'None Provided'}
    Contextual Sample Data (first {sample_rows} rows from `POLICY.csv`):
    ```
    {policy_sample}
    ```

    **Output Format:**
    Provide two separate Markdown tables, one for each target schema (Claim and Policy).

    **Table: Claim (Target Schema)**
    | Target Column Name (PMT) | Inferred Data Type | Description (Based on Name and Sample Data Context) |
    |---|---|---|
    { '| ... | ... | ... |' if claim_target_cols else '| (No target columns defined in schema) | - | - |'}

    **Table: Policy (Target Schema)**
    | Target Column Name (PMT) | Inferred Data Type | Description (Based on Name and Sample Data Context) |
    |---|---|---|
    { '| ... | ... | ... |' if policy_target_cols else '| (No target columns defined in schema) | - | - |'}

    Instructions:
    1. Focus EXCLUSIVELY on describing the TARGET columns listed in the 'Definitive Target Columns' sections.
    2. Infer the Data Type by considering both the column name and the patterns/values seen in the corresponding sample data (if applicable). Use common types like Integer, String, Date, Datetime, Float/Decimal, Boolean/Flag.
    3. Write a concise Description for each TARGET column, explaining its likely purpose based on its name and the sample data context.
    4. Ensure ALL Definitive Target Columns are listed in the output tables.
    5. Start the output directly with the first Markdown table header (`**Table: Claim (Target Schema)**`). Do not add introductory text.
    """)

def create_csv_to_schema_mapping_prompt(
    csv_filepath,
    actual_csv_cols,
    target_schema_cols, # Keep this for listing unmatched target cols if dict is partial
    sample_data,
    sample_rows,
    table_name,
    target_dictionary_details=None # NEW: Add parameter for dictionary info
):
    """Generates a prompt to map columns from a specific CSV file to its target PMT schema, potentially using a detailed data dictionary."""

    target_section_header = f"TARGET: Schema for '{table_name}' table (defined in `PMT_Schema*.csv`)"
    target_column_list_text = f"Target Schema Columns (Full list): {', '.join(target_schema_cols) if target_schema_cols else 'None Defined'}"

    target_dictionary_text = ""
    instruction_emphasis = ""
    if target_dictionary_details and table_name.lower() == "claim": # Only add if provided and for the 'Claim' table
         target_dictionary_text = f"""
    Authoritative Target Data Dictionary Details ({table_name} Schema - Provided Externally):
    ```
    {target_dictionary_details}
    ```
    **CRITICAL: Use the detailed descriptions and types from THIS authoritative dictionary above to understand the target schema accurately.**
    """
         instruction_emphasis = "**Prioritize the Authoritative Target Data Dictionary Details provided above for understanding the target columns.**"
    else:
        target_dictionary_text = """
    (Detailed target data dictionary not provided for this table; rely primarily on target column names and source sample data context for target understanding.)
    """
        instruction_emphasis = ""

    # Construct the prompt string
    prompt = textwrap.dedent(f"""
    Your task is to create a detailed mapping from the SOURCE columns found in the data file `{csv_filepath}` to the TARGET schema columns defined for the '{table_name}' table. Present this mapping as a single, comprehensive Markdown table.

    **Analysis Context:**

    1.  **SOURCE:** Data File `{csv_filepath}`
        *   Actual Source Columns Found in File: {actual_csv_cols if actual_csv_cols else 'None Found / File Empty'}
        *   Sample Data (first {sample_rows} rows from `{csv_filepath}` for context):
            ```
            {sample_data}
            ```

    2.  **{target_section_header}**
        {target_column_list_text}
        {target_dictionary_text}

    **Mapping Goal:** Map EACH source column to its most likely target column, or indicate if no suitable target exists. Also, identify target columns that do not have a corresponding source. {instruction_emphasis}

    **Output Table Format:**
    Generate ONE SINGLE Markdown table with the following structure:

    | Source Column ({csv_filepath}) | Target Column ({table_name} Schema) | Match Confidence | Justification / Mapping Logic / Transformation Notes |
    |------------------------------------|-----------------------------------|------------------|------------------------------------------------------|
    | [source_csv_column_name]           | [target_schema_column_name]       | High/Medium/Low  | [Specific reason based on source/target names, dictionary description (if provided), sample data. Transformation: e.g., "Requires code mapping", "Needs date formatting"] |
    | [source_csv_column_name]           |                                   | None             | [Reason for no match: e.g., "Source index field", "Legacy field not in target dictionary", "Redundant info"] |
    |                                    | [target_schema_column_name]       | None             | [Reason for no source: e.g., "Target column defined in dictionary but no source found", "Derived field", "Internal ID"] |
    | ...                                | ...                               | ...              | ...                                                  |

    **Detailed Instructions for Analysis and Output:**

    1.  **Direction:** Map FROM the 'Actual Source Columns' TO the 'Target Schema Columns'.
    2.  **Use Dictionary (if provided for '{table_name}'):** Heavily rely on the **Authoritative Target Data Dictionary Details** for the definition, data type, and purpose of each target column. Compare the source column's name and sample data against the target dictionary entry.
    3.  **Systematic Mapping:** For *every* source column, attempt to find the *best* match in the target schema list, informed by the dictionary (if available).
    4.  **Confidence Score:** Assign **High**, **Medium**, **Low**, or **None**. Base this on the clarity of the match between the source column (name/data) and the target column's definition (ideally from the dictionary).
    5.  **Justification & Notes:** Explain the match based on the dictionary description (if available) and source context. Note required transformations. For non-matches (Source or Target), explain *why* based on comparison with the dictionary/source data.
    6.  **Comprehensive Coverage:** Ensure *every* source column appears exactly once in the 'Source Column' field. Ensure *every* target column (from the schema list/dictionary) appears at least once (either matched to a source or listed as unmatched). List each unmatched target column *once*.
    7.  **Format:** Strictly adhere to the single Markdown table format. Start output *directly* with the table header (`| Source Column...`). No introductory text.

    Generate the complete mapping table based on this analysis.
    """)
    return prompt


# --- AI Execution Function ---
def run_ai_prompt(prompt_text, title):
    """Sends prompt to the AI model and returns the response text."""
    print(f"\n--- {title} ---")
    print(f"Sending prompt to model '{MODEL_NAME}'...")
    if not prompt_text or not prompt_text.strip():
        print("Warning: Prompt is empty. Skipping AI call.")
        return "Error: Prompt was empty."
    try:
        request_options = {"timeout": 600} # 10 minutes timeout
        response = model.generate_content(prompt_text, request_options=request_options)
        response_text = None

        if hasattr(response, 'text'):
             response_text = response.text
        elif response.parts:
             response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        if response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             print(f"Warning: Response was blocked by safety settings. Reason: {block_reason}")
             return f"Response blocked due to: {block_reason}"
        elif not response_text:
             finish_reason = 'UNKNOWN'; safety_ratings = 'N/A'
             if response.candidates:
                 finish_reason = str(response.candidates[0].finish_reason)
                 if hasattr(response.candidates[0], 'safety_ratings'):
                    safety_ratings = str(response.candidates[0].safety_ratings)
             print(f"Warning: Received an empty or incomplete response from the AI. Finish reason: {finish_reason}, Safety Ratings: {safety_ratings}")
             return f"Received an empty/incomplete response from the AI (Finish Reason: {finish_reason})."

        print("Response received.")
        if ipython_display_available: display(Markdown(response_text))
        else: print(response_text)
        return response_text
    except Exception as e:
        print(f"\nERROR: An error occurred during the API call: {e}")
        if "timeout" in str(e).lower(): print("The request may have timed out. Consider increasing the timeout in 'request_options'.")
        print("Check API key, network, rate limits, model access, prompt content, and quotas.")
        return f"ERROR during API call: {e}"

# --- Output Saving Function ---
def save_output(filename, content):
    """Saves content to a file, checking for errors/blocking messages first."""
    if content and "Response blocked" not in content and "empty/incomplete response" not in content and "ERROR during API call" not in content and "Error: Prompt was empty" not in content :
        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write(content)
            print(f"\nOutput saved to {filename}")
        except Exception as e:
            print(f"\nError saving file {filename}: {e}")
    else:
        print(f"\nSkipping saving {filename} due to blocked/empty/incomplete response, API error, or empty prompt.")
        # print(f"Content not saved:\n---\n{content}\n---") # Uncomment for debugging


# --- Main Execution Logic ---

print("\n--- Reading Input Data ---")
# 1. Read the TARGET schema columns from PMT_Schema file
claim_target_schema_cols, policy_target_schema_cols = read_pmt_schema(PMT_SCHEMA_CSV_PATH)

# 2. Read ACTUAL columns and sample data from the data CSV files
claims_actual_csv_cols, claims_sample_data = read_csv_data_info(CLAIMS_DATA_CSV_PATH, SAMPLE_ROWS_TO_INCLUDE)
policy_actual_csv_cols, policy_sample_data = read_csv_data_info(POLICY_DATA_CSV_PATH, SAMPLE_ROWS_TO_INCLUDE)

# --- Load or Define Authoritative Claim Dictionary Details ---
# Using the snippet defined earlier. Replace this with file loading if you pre-parse.
authoritative_claim_dict_str = claim_dictionary_details_snippet
# authoritative_claim_dict_str = None # Set to None if you don't have it

# 3. Basic checks before proceeding
if not claim_target_schema_cols and not policy_target_schema_cols:
    print("\nERROR: No TARGET columns found for either 'Claim' or 'Policy' tables in the PMT Schema file.")
    sys.exit(1)
if not claims_actual_csv_cols and not policy_actual_csv_cols:
     print("\nERROR: Could not read actual columns from either data CSV file.")
     sys.exit(1)


print("\n--- Generating AI Prompts ---")
# 4a. Create Data Dictionary prompt (describes TARGET schema)
prompt_dd = create_data_dictionary_prompt(
    claim_target_schema_cols, claims_sample_data,
    policy_target_schema_cols, policy_sample_data,
    SAMPLE_ROWS_TO_INCLUDE
)

# 4b. Create Claim CSV -> Claim Schema mapping prompt (WITH dictionary)
prompt_claim_map = None
if claims_actual_csv_cols and claim_target_schema_cols:
    prompt_claim_map = create_csv_to_schema_mapping_prompt(
        csv_filepath=CLAIMS_DATA_CSV_PATH,
        actual_csv_cols=claims_actual_csv_cols,
        target_schema_cols=claim_target_schema_cols,
        sample_data=claims_sample_data,
        sample_rows=SAMPLE_ROWS_TO_INCLUDE,
        table_name="Claim",
        target_dictionary_details=authoritative_claim_dict_str # Pass the dictionary details
    )
else:
    print(f"\nWarning: Skipping Claim mapping prompt generation due to missing actual/target columns.")

# 4c. Create Policy CSV -> Policy Schema mapping prompt (WITHOUT specific dictionary)
prompt_policy_map = None
if policy_actual_csv_cols and policy_target_schema_cols:
    prompt_policy_map = create_csv_to_schema_mapping_prompt(
        csv_filepath=POLICY_DATA_CSV_PATH,
        actual_csv_cols=policy_actual_csv_cols,
        target_schema_cols=policy_target_schema_cols,
        sample_data=policy_sample_data,
        sample_rows=SAMPLE_ROWS_TO_INCLUDE,
        table_name="Policy",
        target_dictionary_details=None # No specific dictionary provided for Policy
    )
else:
    print(f"\nWarning: Skipping Policy mapping prompt generation due to missing actual/target columns.")


print("\n--- Executing AI Prompts ---")
# 5. Run the prompts through the AI model
data_dictionary_output = run_ai_prompt(prompt_dd, "Generating Data Dictionary for Target Schema")
claim_mapping_output = run_ai_prompt(prompt_claim_map, f"Generating Mapping: {CLAIMS_DATA_CSV_PATH} -> Claim Schema")
policy_mapping_output = run_ai_prompt(prompt_policy_map, f"Generating Mapping: {POLICY_DATA_CSV_PATH} -> Policy Schema")


print("\n--- Saving Outputs ---")
# 6. Save the valid AI responses to markdown files
save_output("target_schema_data_dictionary.md", data_dictionary_output)
save_output("claim_csv_to_schema_mapping_with_dict.md", claim_mapping_output) # Renamed output
save_output("policy_csv_to_schema_mapping.md", policy_mapping_output)

print("\n--- Script Finished ---")
