from datetime import datetime

# import en_core_web_sm
# import pandas as pd
import pytz

# NLP_DISABLE = ["tagger", "parser", "ner"]

# nlp = en_core_web_sm.load(disable=NLP_DISABLE)


def check_tweet(status) -> bool:
    '''Return True if tweet satisfies specific criteria
    '''
    return all([
        ('text' in status),
        (status['coordinates']),
        (status['user']['friends_count'] > 0),  # following
        (status['user']['followers_count'] > 0),  # followers
    ])


def date_string_to_datetime(
    date_string: str,
    fmt: str = '%a %b %d %H:%M:%S +0000 %Y',
    tzinfo=pytz.UTC,
) -> datetime:
    return datetime.strptime(date_string, fmt).replace(tzinfo=tzinfo)


# def stopword_lemma(X: pd.DataFrame, col: str) -> pd.DataFrame:
#     def _token_filter(token):
#         return not (token.is_punct | token.is_space | token.is_stop | (token.lemma_ == '-PRON-') | (len(token.text) <= 1))

#     Xa = X.copy()

#     for doc, idx in nlp.pipe(((text, i) for i, text in Xa[col].fillna('').iteritems()), as_tuples=True, n_process=1):
#         tokens = [token.lemma_ for token in doc if _token_filter(token)]
#         Xa.loc[idx, col] = ' '.join(tokens)

#     return Xa
