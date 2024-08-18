# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)
Root_dir = os.path.dirname(__file__)

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)

        # yyy = mergeable_ranks['آب']
        # yyy = mergeable_ranks[100680]

        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        
    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        # assert isinstance(s, (str, np.str_))

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# llama3_tokenizer = Tokenizer(model_path=Root_dir+"\\tokenizer.model")

# llama3_tokenizer = Tokenizer(model_path=Root_dir+"\\tokenizer.model")

# itest = llama3_tokenizer.encode(' آب', bos=False, eos=False)

# itest = llama3_tokenizer.encode('آنها میتوانند با شما دوستی کنند ولی اگر کارگران در شرابخانه آبادانان را بگذارند', bos=False, eos=False)
# print(itest)

# import json
# import random


# def cleanword(word):
#      word = word.replace('ّ','')
#      word = word.replace('ِ','') 
#      word = word.replace('ُ','')
#      word = word.replace('َ','')
#      word = word.replace('ً','')
#      word = word.replace('ٌ','')
#      word = word.replace('ٍ','')
#      word = word.replace('،','')
#      word = word.replace('؛','')
#      word = word.replace(',','')
#      word = word.replace(']','')
#      word = word.replace('[','')
#      word = word.replace('\\','')
#      word = word.replace('}','')
#      word = word.replace('{','')
#      word = word.replace('(','')
#      word = word.replace(')','')
#      word = word.replace('*','')
#      word = word.replace('&','')
#      word = word.replace('.','')
#     #  word = word.replace(' ','')
#      word = word.replace('ﷺ','')
#      word = word.replace('»','')
#      word = word.replace('»','')
#      word = word.replace('\t','')
#      word = word.replace('/n','')
#      word = word.replace('؟','')
#      word = word.replace('>','')
#      word = word.replace('<','')
#      word = word.replace('S1','')
#      word = word.replace('S2','')
#      word = word.replace('E','')
#      word = word.replace('Y','')
#      word = word.replace('ي','ی')
#      word = word.replace('ئ','ی')
#      word = word.replace('ة','ه')
#      word = word.replace('هٔ','ه')
#      word = word.replace('—','')
#      word = word.replace(':','')
#      word = word.replace('«','')
#      word = word.replace('-','')
#      word = word.replace('"','')
#      word = word.replace('_','')
#      word = word.replace('!','')
#      word = word.replace('/','')
#      word = word.replace('1','')
#      word = word.replace('2','')
#      word = word.replace('3','')
#      word = word.replace('4','')
#      word = word.replace('5','')
#      word = word.replace('6','')
#      word = word.replace('7','')
#      word = word.replace('8','')
#      word = word.replace('9','')
#      word = word.replace('0','')
#      word = word.replace('=','')
#      word = word.replace('÷','')
#      word = word.replace('۰','')
#      word = word.replace('۱','')
#      word = word.replace('۲','')
#      word = word.replace('۳','')
#      word = word.replace('۴','')
#      word = word.replace('۵','')
#      word = word.replace('۶','')
#      word = word.replace('۷','')
#      word = word.replace('۸','')
#      word = word.replace('۹','')
#      word = word.replace('…','')
#      word = word.replace('٭','')
#      word = word.replace('“','')
#      word = word.replace('”','')
#      word = word.replace('–','')
#      word = word.replace('‎','')
#      word = word.replace('‏','')
#      word = word.replace('‬','')      
#      return word
     

# def process_shard(shard_index, shard_filename):
#     # create tokenizer and encode function within the process
#     def encode(x):
#         return llama3_tokenizer.encode(x, bos=True, eos=True)

#     with open(shard_filename, "r", encoding='utf-8') as f:
#         data = json.load(f)
#     rng = random.Random(1337 + shard_index)
#     rng.shuffle(data)
#     all_tokens = []
#     for example in data:

#         text = example["sher"]
#         text = text.strip()  # get rid of leading/trailing whitespace
#         # text = cleanword(text)
#         tokens = encode(text)
#         all_tokens.extend(tokens)
#     return all_tokens


# DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "textData")
# data_dir = os.path.join(DATA_CACHE_DIR, "Data-text")
# shard_filenames = os.path.join(data_dir, "abdullah.json")

# mytttoken = process_shard(1,shard_filenames)

# curenttoken = llama3_tokenizer.decode(mytttoken)

# import Comon as com
# com.filecontrol.SaveText(curenttoken,"Hafezfull2_decode.txt")
# print(mytttoken)
# for i in range(0,len(itest),1) :
#     curenttoken = llama3_tokenizer.decode(itest[0:i])
#     print(curenttoken,itest[0:i])

