from mrjob.job import MRJob
from mrjob.step import MRStep
import logging
import re
import spacy

nlp = spacy.load("en_core_web_sm")


class MRCommonKeyWords(MRJob):
    """
    Mapper returns tuples as ('Genre', 'Title').
    Title is first filtered to get only relevant keywords.
    We can then count most recurents words for each genre
    """

    def configure_args(self):
        super(MRCommonKeyWords, self).configure_args()
        logging.basicConfig(level=logging.INFO)

    def mapper_get_genre_title(self, _, line):
        """
        Yields unique combination of (genre, word) 1.
        Each word in the title is associated each genre the movie is attached to.
        Combined afterwards
        """

        # Generates list with [id, Title (year), Genres]
        cols = line.split(",")

        genres = cols[-1]
        title = " ".join(cols[1:-1])
        # removes the trailing (year)
        title_no_year = re.sub(r"\(\d{4}\)", "", title)
        nlp_title = nlp(title_no_year)
        # remove punctuation aux verbs, determinants, ... and set to all lower case
        keywords = [
            token.text.lower()
            for token in nlp_title
            if not token.is_stop
            and not token.is_punct
            and token.pos_ not in ("AUX", "DET", "ADP", "CCONJ")
        ]
        keywords = [word.strip() for word in keywords if word.strip()]
        # splited_w = title_no_year.split()
        # logging.info(f"keywords: {splited_w}")

        for genre in genres.split("|"):
            for word in keywords:
                yield (str(genre), word), 1

    def reducer_words(self, key, count):
        """
        Key is of type (genre, word).
        Sums all instance of unique (genre, word)
        """
        genre, word = key
        yield genre, (word, sum(count))

    def reducer_find_max_10(self, genre, word_count):
        top_10 = sorted(word_count, reverse=True, key=lambda x: x[1])[:10]
        for word, count in top_10:
            yield genre, (word, count)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_genre_title,
                   reducer=self.reducer_words),
            MRStep(reducer=self.reducer_find_max_10),
        ]


if __name__ == "__main__":
    MRCommonKeyWords.run()
