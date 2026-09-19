"""Cornell Movie-Dialogs corpus: download, wrangle, vocabulary, batching.

Ported from the "Preparing Data" section of the official PyTorch chatbot
tutorial (https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html).
"""

from __future__ import annotations

import ast
import itertools
import os
import re
import unicodedata
import urllib.request
import zipfile

import torch

MAX_LENGTH = 10  # Maximum sentence length to consider for the model
MIN_COUNT = 3    # Minimum word count threshold for trimming

CORPUS_URL = os.environ.get(
    "TILECHAT_CORPUS_URL",
    "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
)
DEFAULT_DATA_DIR = os.environ.get(
    "TILECHAT_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    """Simple word <-> index vocabulary (from the tutorial)."""

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print(
            "keep_words {} / {} = {:.4f}".format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            )
        )
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)


# Turn a Unicode string to plain ASCII (https://stackoverflow.com/a/518232/2809427)
def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_corpus(data_dir=None):
    """Download + extract the Cornell Movie-Dialogs corpus if needed.

    Returns the path to the raw ``movie_lines.txt``.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    corpus_dir = os.path.join(data_dir, "cornell movie-dialogs corpus")
    lines_path = os.path.join(corpus_dir, "movie_lines.txt")
    if not os.path.exists(lines_path):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, "cornell_movie_dialogs_corpus.zip")
        if not os.path.exists(zip_path):
            print("Downloading Cornell Movie-Dialogs Corpus ...")
            urllib.request.urlretrieve(CORPUS_URL, zip_path)
        print("Extracting corpus ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    if not os.path.exists(lines_path):
        raise FileNotFoundError(f"corpus not found at {lines_path}")
    return lines_path


def loadLines(fileName, fields):
    """Split each line of the file into a dictionary of fields."""
    lines = {}
    with open(fileName, encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj["lineID"]] = lineObj
    return lines


def loadConversations(fileName, lines, fields):
    """Groups fields of lines from ``loadLines`` into conversations."""
    conversations = []
    with open(fileName, encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert the last field (a string with a python list of utterance
            # IDs) into a real list.  ast.literal_eval is a safe version of the
            # tutorial's eval().
            lineIds = ast.literal_eval(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractSentencePairs(conversations):
    """Extracts pairs of sentences from conversations."""
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):  # ignore last line (no answer)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            if inputLine and targetLine:  # filter wrong samples (if one list is empty)
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def prepare_formatted_pairs(data_dir=None):
    """The tutorial's data-wrangling step: raw corpus -> formatted_movie_lines.txt."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    corpus_dir = os.path.join(data_dir, "cornell movie-dialogs corpus")
    datafile = os.path.join(data_dir, "formatted_movie_lines.txt")
    if os.path.exists(datafile):
        return datafile
    lines_path = ensure_corpus(data_dir)
    print("Processing corpus into conversation pairs ...")
    lines = loadLines(
        lines_path,
        ["lineID", "characterID", "movieID", "character", "text"],
    )
    conversations = loadConversations(
        os.path.join(corpus_dir, "movie_conversations.txt"),
        lines,
        ["character1ID", "character2ID", "movieID", "utteranceIDs"],
    )
    qa_pairs = extractSentencePairs(conversations)
    with open(datafile, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(normalizeString(pair[0]) + "\t" + normalizeString(pair[1]) + "\n")
    return datafile


def readVocs(datafile, corpus_name):
    print("Reading and preprocessing file ...")
    lines = open(datafile, encoding="utf-8", errors="ignore").read().strip().split("\n")
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(p):
    # Input sequences need to preserve the last word for the EOS token
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def loadPrepareData(data_dir=None, corpus_name="cornell movie-dialogs corpus"):
    """Load, wrangle and index the corpus. Returns (voc, pairs)."""
    datafile = prepare_formatted_pairs(data_dir)
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words ...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, min_count=MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in pair[0].split(" "):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in pair[1].split(" "):
            if word not in voc.word2index:
                keep_output = False
                break
        # Only keep pairs that do not contain trimmed words
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print(
        "Trimmed from {} pairs to {}, {:.4f} of total".format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)
        )
    )
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_length = max(len(indexes) for indexes in indexes_batch)
    padList = zeroPadding(indexes_batch)
    mask = torch.BoolTensor(binaryMatrix(padList))
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_length


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_length = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_length
