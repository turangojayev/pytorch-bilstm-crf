from sklearn.utils import shuffle
from tqdm import tqdm

from data_utils import CoNLLDataset, get_vocabs, UNK, NUM, PAD, \
    get_embedding_vocab, write_vocab, get_char_vocab, filter_embeddings_in_vocabulary, get_processing_word, get_df, \
    load_vocab_vectors, load_vocab, STOP_TAG, START_TAG, TOKEN2IDX


def main():
    words_file = "../data/words.txt"
    tags_file = "../data/tags.txt"
    chars_file = "../data/chars.txt"
    test_file = '../data/more_annotated_test.conll'
    train_file = '../data/more_annotated_train.conll'
    embeddings_file = "/usr/data/audi/w2v/word2vec.txt"
    #embeddings_file = "C:\\Users\\turan\\Downloads\\audi_w2v\\word2vec.txt"
    filtered_embeddings_file = "../data/filtered_embeddings.txt"

    processing_word = get_processing_word(lowercase=False)

    test = CoNLLDataset(test_file, processing_word)
    train = CoNLLDataset(train_file, processing_word)

    vocab_words, vocab_tags = get_vocabs([train, test])
    embedding_vocab = get_embedding_vocab(embeddings_file)

    vocab = vocab_words & embedding_vocab
    print('{} overlapping words'.format(len(vocab)))
    vocab.add(UNK)
    vocab.add(NUM)
    vocab = list(vocab)
    vocab.insert(TOKEN2IDX[PAD], PAD)
    vocab.insert(TOKEN2IDX[START_TAG], START_TAG)
    vocab.insert(TOKEN2IDX[STOP_TAG], STOP_TAG)
    print(len(vocab))

    write_vocab(vocab, words_file)
    write_vocab(vocab_tags, tags_file)

    filter_embeddings_in_vocabulary(words_file, embeddings_file, filtered_embeddings_file)

    # train = CoNLLDataset(train_file)
    # vocab_chars = get_char_vocab(train)
    vocab_chars = get_char_vocab(vocab_words)
    write_vocab(vocab_chars, chars_file)


def helper():
    import numpy
    import pandas

    df = get_df()
    print(df.shape)
    df = df.drop_duplicates().reset_index(drop=True)
    print(df.shape)

    entity2count = dict()
    for entities in df.entities:
        for _, _, _, entity in entities:
            entity = entity.lower()
            if entity not in entity2count:
                entity2count[entity] = 0
            entity2count[entity] += 1

    print('{} entities'.format(len(entity2count)))
    # print(entity2count)
    random_entity_count = 200
    random_ents = numpy.random.choice(list(entity2count.keys()), random_entity_count, False)
    summ = 0

    for ent in random_ents:
        summ += entity2count[ent]
    print('{} occur totally {}'.format(random_entity_count, summ))

    random_ents = set(random_ents)
    print(random_ents)
    test_indices = set()
    for i, row in df.iterrows():
        for _, _, _, entity in row.entities:
            entity = entity.lower()
            if entity in random_ents or \
                    (entity + 's') in random_ents or \
                    (entity + 'e') in random_ents or \
                    (entity + 'n') in random_ents or \
                    (entity + 'en') in random_ents or \
                    entity[:-1] in random_ents:
                test_indices.add(i)

    test_indices = list(test_indices)
    test_indices = numpy.asarray(test_indices)

    df_test = df.iloc[test_indices]
    merged = df.merge(df_test, indicator=True, how='outer')
    df_train = merged[merged._merge == 'left_only']
    df_train.drop(columns=['_merge'], inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    print('{} without entities'.format(len(df_train[df_train.types == ()])))
    without = numpy.random.choice(df_train[df_train.types == ()].index.values, 400, replace=False)
    empty_for_test = df_train.iloc[without]
    merged = df_train.merge(empty_for_test, indicator=True, how='outer')
    df_train = merged[merged._merge == 'left_only']
    df_train.drop(columns=['_merge'], inplace=True)
    df_train.reset_index(drop=True)
    df_test = pandas.concat([df_test, empty_for_test])
    df_test.reset_index(drop=True)
    print(df_train.shape, df_test.shape)
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    _write_to_conll_format('data/more_annotated_train.conll', df_train)
    _write_to_conll_format('data/more_annotated_test.conll', df_test)


def _write_to_conll_format(path, df):
    def _get_instance(row):
        result = []
        data = row.to_dict()
        entities = data['entities']
        tokens = data['texts'].split()
        if not entities:
            result = ['{} O'.format(t) for t in tokens]
        else:
            idx, entityIdx = 0, 0

            while idx < len(tokens) and entityIdx < len(entities):
                if idx < entities[entityIdx][0]:
                    result.append('{} O'.format(tokens[idx]))
                elif idx == entities[entityIdx][0]:
                    result.append('{} B-{}'.format(tokens[idx], entities[entityIdx][2]))
                elif entities[entityIdx][0] < idx < entities[entityIdx][1]:
                    result.append('{} I-{}'.format(tokens[idx], entities[entityIdx][2]))
                elif idx == entities[entityIdx][1]:
                    # result.append('{} O'.format(tokens[idx]))
                    entityIdx += 1
                    continue

                idx += 1

            while idx < len(tokens):
                result.append('{} O'.format(tokens[idx]))
                idx += 1
        return result

    with open(path, 'wt') as f:
        for i, row in tqdm(df.iterrows()):
            lines = _get_instance(row)
            for line in lines:
                print(line, file=f)
            print('', file=f)


if __name__ == "__main__":
    main()
    # helper()
    embeddings = load_vocab_vectors("../data/filtered_embeddings.txt")
    #print(embeddings.shape)
    #print(embeddings[0])
    #print(embeddings[load_vocab("../data/words.txt")[UNK]])
    #print(embeddings[load_vocab("../data/words.txt")[NUM]])
    #print(embeddings[load_vocab("../data/words.txt")[START_TAG]])
    #print(embeddings[load_vocab("../data/words.txt")[STOP_TAG]])
