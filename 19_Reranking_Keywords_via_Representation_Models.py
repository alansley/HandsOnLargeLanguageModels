# Note: This one starts of like `18a_Topic_Modeling_via_Embedding_then_Dimensional_Reduction` but then we add a final
# block to the pipeline to re-rank keywords using better techniques than the default rather than doing just
# the clustering we use BERTopic and class-based-TF-IDF (Term-Frequency / Inverse-Document-Frequency) analysis to
# generate the topics of the clusters. See p146-155.
#
# The full pipline goes:
# 1.) Clustering:
#     a.) SBERT
#     b.) UMAP
#     c.) HDBSCAN
# 2.) Topic Representation:
#     a.) CountVectorizer (to generate the term frequency (TF))
#     b.) c-TF-IDF
# 3.) Re-ranking Keywords                                          <------ NEW!
#         - First we use a KeyBERTInspired representation model
#         - Then we use a Maximal Marginal Relevance (MMR) model just to see the difference,
#         - Then we use the MMR method again with a larger T5 model to show how it improves the topic analysis.

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import TextGeneration
from copy import deepcopy
from datasets import load_dataset, Dataset, load_from_disk
from hdbscan import HDBSCAN
import os
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from umap import UMAP  # Note: We actually install the "umap-learn" package rather than just "umap" - see requirements.txt

print("IMPORTANT: This does a TON of number crunching and may take a few minutes, even on CUDA with a RTX 4090.")
print("We print progress, try to save models on first run to speed up further runs, but as mentioned - serious number crunching ahead.")

# Load data from Hugging Face
dataset_name = "maartengr/arxiv_nlp"
slice_name = "train"
cached_dataset_uri = "DataFiles/" + dataset_name

def load_data(slice_name_arg: str = None):
    # Get the data from hugging face if we don't already have it then save it to our cache
    # Note: If a specific slice was specified then we only grab that slice.
    if not os.path.exists(cached_dataset_uri):
        if slice_name is not None:
            data: Dataset = load_dataset(dataset_name)[slice_name_arg]
        else:
            data: Dataset = load_dataset(dataset_name)
        data.save_to_disk(cached_dataset_uri)
        return data
    else:
        # Otherwise if we already have a cached copy then load that rather than re-downloading.
        # Note: If we previously only downloaded a specific slice we don't have to specify it as that's all we have!
        data: Dataset = load_from_disk(cached_dataset_uri)
        return data

# Download the dataset on first run, then save it & use the cached version in future runs
dataset: Dataset = load_data(slice_name)

# Extract data
titles    = dataset["Titles"]
abstracts = dataset["Abstracts"]

# Create an embedding for each abstract.
# Note: Generating these embeddings takes ~1 min on a RTX 4090 so will likely take several minutes if running CPU-only!
embedding_model_name = "thenlper/gte-small"
embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
print("\n--- Encoding abstracts as embeddings...")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device="cuda")

# Show that we got 44,949 abstracts each using an embedding of 384 values
print(f"\n--- Created embeddings for {embeddings.shape[0]} abstracts, where each embedding has {embeddings.shape[1]} values.")

# Reduce 384-dimensional embeddings to 5 dimensions.
# Note: Generally, values between 5 and 10 work well to capture high-dimensional global structures.
reduced_dimension_count = 5
print(f"\n--- Creating UMAP model to reduce dimensions from {embeddings.shape[1]} to {reduced_dimension_count}...")
umap_model = UMAP(
    n_components=reduced_dimension_count,
    min_dist=0.0,

    # Note: We could use `cosine` here - but always keep it `cosine->cosine` or `euclidean->euclidean` wrt our UMAP
    # model and HDBSCAN model (coming next).
    metric="euclidean",
    random_state=42,
    verbose=True
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Now use HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to perform clustering.
# As a density-based method, HDBSCAN can also detect outliers in the data, which are data points that do not belong to
# any cluster. These outliers will not be assigned or forced to belong to any cluster. In other words, they are ignored.
# Since our ArXiv articles might contain some niche papers, using a model that detects outliers could be helpful.

# We fit the model and extract the clusters
print(f"\n--- Creating HDBSCAN model to map embeddings into clusters...")
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom"
)
# Note: I'm just doing the `fit` separately to stop the line below warning be that it doesn't know what `labels_` is
# because the `fit` method returns an Object not a HDBSCAN object.
hdbscan_model.fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate? In this instance we're expecting 152 - which is quite a lot of clusters...
# Note: Because HDBSCAN uses outliers these will be topics 0..150 (e.g., 151 topics) and the special "-1" topic will be
# used for outliers that don't fit cleanly into any of our other topics.
num_clusters = len(set(clusters))
print(f"\n--- HDBSCAN generated {num_clusters} clusters.")

# Train our model with our previously defined models (generate & save on first run, load saved topic-model otherwise)
topic_model = None
keywords_dir = "DataFiles/Keywords"
small_keywords_nlp_uri = keywords_dir + "/bertopic_arxiv_nlp_t5_flan_small"
if os.path.exists(small_keywords_nlp_uri):
    topic_model = BERTopic.load(small_keywords_nlp_uri)
else:
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    ).fit(abstracts, embeddings)
    os.makedirs(keywords_dir, exist_ok=True)  # Make sure the directory exists before saving
    topic_model.save(path=small_keywords_nlp_uri)

# Save original representations
original_topics = deepcopy(topic_model.topic_representations_)

def topic_differences(model, original_topics_arg, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        # Extract top 5 words per topic per model
        og_words  = " | ".join(list(zip(*original_topics_arg[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df

# Update our topic representations using the KeyBERTInspired representation model
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Set pandas options to avoid truncation
pd.set_option('display.max_colwidth', None)        # Don't truncate column contents
pd.set_option('display.max_columns', None)         # Show all columns
pd.set_option('display.width', 0)                  # Automatically detect console width
pd.set_option('display.expand_frame_repr', False)  # Print in one block if possible

# Get the differences between the original and representation-model topics
dataframe: DataFrame = topic_differences(topic_model, original_topics)

# The indices run 0/1/2/3/4 and the topics ALSO run 0/1/2/3/4 so to avoid printing both indices AND topics we set the
# index to BE the topic field.
dataframe.set_index("Topic", inplace=True)

# Now print out the before/after - this should get us details like:
#
# Topic                                         Original                                                            Updated
# 0          speech | asr | recognition | acoustic | end               phonetic | transcription | speech | encoder | spoken
# 1          translation | nmt | machine | bleu | neural  translation | translate | translations | translated | monolingual
# 2        hate | offensive | speech | detection | toxic                hate | hateful | language | offensive | classifiers
# 3      relation | extraction | re | relations | entity         relation | relations | relational | sentences | extracting
# 4        gender | bias | biases | debiasing | fairness                       gender | gendered | bias | biases | pronouns
#
print("\n--- Using KeyBERTInspired gives us keywords with slightly better diversity of their representations:")
print(dataframe)

# Looking at the above, note that topic 1 has a ton of very simulation words:
#
#     translation | translate | translations | translated | monolingual
#
# All the "translate" ones are just plurals, verbs, past-tense etc. of the same word - so let's do better than that by
# using Maximal Marginal Relevance (MMR) to diversify our keywords from one-another but still keep them related to the
# documents.

# Update our topic representations to use MaximalMarginalRelevance (generate & save on first run, load saved topic-model otherwise)
representation_model = MaximalMarginalRelevance(diversity=0.2)
small_keywords_mmr_uri = keywords_dir + "/bertopic_arxiv_mmr_t5_flan_small"
if os.path.exists(small_keywords_mmr_uri):
    topic_model = BERTopic.load(small_keywords_mmr_uri)
else:
    topic_model.update_topics(abstracts, representation_model=representation_model)
    topic_model.save(small_keywords_mmr_uri)

# Show topic differences of original via MMR representations. This gets us output like:
#
# Topic                                         Original                                                  Updated
# 0          speech | asr | recognition | acoustic | end                    speech | asr | wer | training | audio
# 1          translation | nmt | machine | bleu | neural             translation | nmt | bleu | neural | parallel
# 2        hate | offensive | speech | detection | toxic             hate | offensive | toxic | abusive | hateful
# 3      relation | extraction | re | relations | entity  relation | extraction | relations | entities | document
# 4        gender | bias | biases | debiasing | fairness            gender | bias | biases | debiasing | fairness
#
dataframe = topic_differences(topic_model, original_topics)
dataframe.set_index("Topic", inplace=True)
print("\n--- Using Maximal Marginal Relevance (MMR) gives us keywords with even better diversity of their representations:")
print(dataframe)

# let's take a copy of our MMR-based topics to compare and contrast topic labels against our original topic labels later
mmr_topics = deepcopy(topic_model.topic_representations_)

# Now let's generate a specific label for each specific topic using a prompt & a generative transformer
# Note: `[DOCUMENTS]` and `[KEYWORDS]` is automatically filled by TextGeneration when we pass it the prompt
prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, give a short description (3-5 words max) of what this topic is about."""

# Update our topic representations using Flan-T5 small
generator = pipeline("text2text-generation", model="google/flan-t5-small", device="cuda")
representation_model = TextGeneration(generator, prompt=prompt, doc_length=50, tokenizer="whitespace")
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences using our original topics - which will get us output like:
#
# Topic                                         Original                     Updated
# 0          speech | asr | recognition | acoustic | end          Speech recognition
# 1          translation | nmt | machine | bleu | neural  Neural Machine Translation
# 2        hate | offensive | speech | detection | toxic                Science/Tech
# 3      relation | extraction | re | relations | entity         relation extraction
# 4        gender | bias | biases | debiasing | fairness                Science/Tech
dataframe = topic_differences(topic_model, original_topics)
dataframe.set_index("Topic", inplace=True)
print("\n--- Using summarisation of our original topics we get the following topic names:")
print(dataframe)

# Show topic differences using our MMR-enhanced keyword topics - which gets us output like:
#
# Topic                                         Original                             Updated
# 0                        speech | asr | wer | training | audio          Speech recognition
# 1                 translation | nmt | bleu | neural | parallel  Neural Machine Translation
# 2                 hate | offensive | toxic | abusive | hateful                Science/Tech
# 3      relation | extraction | relations | entities | document         relation extraction
# 4                gender | bias | biases | debiasing | fairness                Science/Tech
dataframe = topic_differences(topic_model, mmr_topics)
dataframe.set_index("Topic", inplace=True)
print("\n--- Using summarisation of our MMR-enhanced topics we get the following topic names:")
print(dataframe)

# Okay, "Science/Tech" as a category is doing a lot of heavy lifting here - and it doesn't really match for topics like
# "gender | bias | biases | debiasing | fairness" - the issue is that our t5-flan model is a bit small (~330MB) and just
# doesn't know enough to take a really good stab at it - so let's use t5-flan-large (~3.1GB) instead.
# Model page: https://huggingface.co/google/flan-t5-large

# The default prompt with flan-t5-large can create super-long topics, so we'll try to tighten it up a bit
tight_prompt = """Given the topic's representative documents:
[DOCUMENTS]
And the keywords: '[KEYWORDS]'
Generate a short, specific label (max 5 words) describing the topic."""

# Create the generator with max_new_tokens set so we don't hundreds of chars as a "topic"
# Note: This was an issue with the original book prompt - I've tightened up the prompt to ask for 3-5 words only.
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0,  # Use device=0 for CUDA, or -1 for CPU
    max_new_tokens=20  # Limit the length of generated text to 20 tokens / ~50 chars
)

# Pass the configured pipeline to TextGeneration
representation_model = TextGeneration(
    generator,
    prompt=tight_prompt,
    doc_length=10,  # IMPORTANT: Rather than giving this 50 words we'll just give it 10 to see if it calms down a bit
    tokenizer="whitespace"
)

# Generate our final t5-flan-large topics (generate & save on first run, load saved topic-model otherwise)
large_keywords_mmr_uri = keywords_dir + "/bertopic_arxiv_mmr_t5_flan_large"
if os.path.exists(large_keywords_mmr_uri):
    topic_model = BERTopic.load(large_keywords_mmr_uri)
else:
    topic_model.update_topics(abstracts, representation_model=representation_model)
    topic_model.save(large_keywords_mmr_uri)

dataframe = topic_differences(topic_model, mmr_topics)
dataframe.set_index("Topic", inplace=True)
print("\n--- Using summarisation of our MMR-enhanced topics using our tight-prompt and the t5-flan-large model we get the following topic names:")
print(dataframe)

# Just for comedy:
#
# When using the original prompt, take a look at topic 3 when using t5-flan-large WITHOUT limiting the max new tokens!
# Hardly succinct!
#
#                                                       Original                                                                                                                                                                                                                                                                            Updated
# Topic
# 0                        speech | asr | wer | training | audio                                                                                                                                                                                Automatic Speech Recognition with Synthetic Audio for End-to-End Text-to-Speech Systems |  |  |  |
# 1                 translation | nmt | bleu | neural | parallel                                                                                                                                                                                                   a hybrid search for attention-based neural machine translation (nmt) |  |  |  |
# 2                 hate | offensive | toxic | abusive | hateful                                                                                                                                                                                 HateXplain: a dataset for automatic detection of hate speech in online social networks |  |  |  |
# 3      relation | extraction | relations | entities | document  Joint entity and relation extraction framework: a unified model to perform entity recognition and relation extraction simultaneously, which can exploit the dependency between the two tasks to mitigate the error propagation problem suffered by the pipeline model |  |  |  |
# 4                gender | bias | biases | debiasing | fairness
