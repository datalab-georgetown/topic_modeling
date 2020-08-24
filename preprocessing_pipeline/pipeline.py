from inspect import signature


class Preprocess:
    name = ''
    id = -1
    total_documents = 0
    freq = dict()
    vocabulary = set()
    dataset_methods = []
    document_methods = []
    adhoc_methods = []

    def remove_empty(self, D):
        new_dataset = []
        for d in D:
            if len(d) > 0:
                new_dataset.append(d)
        return new_dataset

    def record_frequencies(self, d):
        '''
        freq = [df, norm_f, max_f]
        '''
        self.total_documents += 1
        for term in set(d):
            if term in self.freq:
                self.freq[term][0] += 1
            else:
                self.freq[term] = [1, 0, 0]
        for term in d:
            self.vocabulary.add(term)
            if term in self.freq:
                self.freq[term][1] += 1 / len(d)
                self.freq[term][2] += 1
            else:
                self.freq[term] = [1, 1 / len(d), 1]

    def flush_frequencies(self):
        self.freq = dict()
        self.vocabulary = set()

    def clean_document_rm(self, d):
        starting_chars = len(''.join(d))
        starting_tokens = len(d)
        for fn_tup in self.document_methods:
            fn = fn_tup[0]
            if len(fn_tup) > 2:
                extra_args = fn_tup[2]
                d = fn(d, **extra_args)
            else:
                d = fn(d)
        ending_chars = len(''.join(d))
        ending_tokens = len(d)
        return [d, starting_chars, starting_tokens, ending_chars,
                ending_tokens]

    def clean_documents_rm(self, D):
        D_prime = []
        D_prime.extend(D)
        for i in range(0, len(D_prime)):
            D_prime[i] = self.clean_document_rm(D_prime[i])
        temp_D_prime = [d[0] for d in D_prime]
        for fn_tup in self.dataset_methods:
            fn = fn_tup[0]
            if len(fn_tup) > 1:
                extra_args = fn_tup[2]
                temp_D_prime = fn(temp_D_prime, **extra_args)
            else:
                temp_D_prime = fn(temp_D_prime)
        for i in range(0, len(D_prime)):
            new_d = temp_D_prime[i]
            D_prime[i][0] = new_d
            D_prime[i][3] = len(''.join(new_d))
            D_prime[i][4] = len(new_d)
        for i in range(0, len(D_prime)):
            self.record_frequencies(D_prime[i][0])
        return D_prime

    def clean_documents_adhoc_rm(self, D):
        for fn_tup in self.adhoc_methods:
            temp_D = [d[0] for d in D]
            fn = fn_tup[0]
            if len(fn_tup) > 1:
                extra_args = fn_tup[1]
                params = signature(fn)
                for param in params.parameters:
                    if param == 'total_documents':
                        extra_args['total_documents'] = self.total_documents
                    elif param == 'freq':
                        extra_args['freq'] = self.freq
                temp_D = fn(temp_D, **extra_args)
            else:
                temp_D = fn(temp_D)
            for i in range(0, len(D)):
                new_d = temp_D[i]
                D[i][0] = new_d
                D[i][3] = len(''.join(new_d))
                D[i][4] = len(new_d)
        return D

    def clean_document(self, d):
        for fn_tup in self.document_methods:
            fn = fn_tup[0]
            if len(fn_tup) > 2:
                extra_args = fn_tup[2]
                d = fn(d, **extra_args)
            else:
                d = fn(d)
        return d

    def clean_documents(self, D):
        for fn_tup in self.dataset_methods:
            fn = fn_tup[0]
            if len(fn_tup) > 1:
                extra_args = fn_tup[2]
                D = fn(D, **extra_args)
            else:
                D = fn(D)
        for i in range(0, len(D)):
            D[i] = self.clean_document(D[i])
            self.record_frequencies(D[i])
        D = self.remove_empty(D)
        return D

    def clean_documents_adhoc(self, D):
        for fn_tup in self.adhoc_methods:
            fn = fn_tup[0]
            if len(fn_tup) > 1:
                extra_args = fn_tup[1]
                params = signature(fn)
                for param in params.parameters:
                    if param == 'total_documents':
                        extra_args['total_documents'] = self.total_documents
                    elif param == 'freq':
                        extra_args['freq'] = self.freq
                D = fn(D, **extra_args)
            else:
                D = fn(D)
        return D
