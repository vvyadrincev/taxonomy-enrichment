#!/usr/bin/env python
# coding: utf-8

import os
import os.path as fs
import lxml.etree as ET
import collections
import logging

def load_tree(function):
    def _load_xml_from_file(*args, **kwargs):
        filepath = kwargs['filepath']
        logging.debug ("Trying to load tree from %s" % (filepath))
        try:
            with open(filepath, 'r') as f:
                function(tree=ET.fromstring(f.read()), *args, **kwargs)
        except Exception as e:
            raise RuntimeError (e)

    return _load_xml_from_file


class RuWordNet(object):
    def __init__(self, load_dir=None):

        if load_dir is None:
            return
        elif fs.isdir(load_dir):
            self.load_fom_dir(load_dir)
            logging.info(f"RuWordNet loaded from {load_dir}")


    @load_tree
    def _load_senses(self, *args, **kwargs):
        if not hasattr(self, 'senses_list'):
            self.senses_list = list()

        self.senses_list.extend([e.attrib
                                 for e in kwargs['tree'].xpath('./sense')])

    @load_tree
    def _load_synset_relations(self, *args, **kwargs):
        if not hasattr(self, 'synset_relations_list'):
            self.synset_relations_list = list()

        self.synset_relations_list.extend([e.attrib
                                           for e in kwargs['tree'].xpath('./relation')])

    @load_tree
    def _load_synsets(self, *args, **kwargs):
        if not hasattr(self, 'synsets_list'):
            self.synsets_list = list()

        self.synsets_list.extend([e.attrib
                                  for e in kwargs['tree'].xpath('./synset')])

    @load_tree
    def _load_composed_of(self, *args, **kwargs):
        if not hasattr(self, 'composed_of_list'):
            self.composed_of_list = list()

        self.composed_of_list.extend([(e.attrib, [s.attrib
                                                  for s in e.xpath('./composed_of/sense')
                                                  ]
                                       )
                                      for e in kwargs['tree'].xpath('./sense')])

    @load_tree
    def _load_derived_from(self, *args, **kwargs):
        if not hasattr(self, 'derived_from_list'):
            self.derived_from_list = list()

        self.derived_from_list.extend([(e.attrib, [s.attrib
                                                  for s in e.xpath('./derived_from/sense')
                                                  ]
                                       )
                                      for e in kwargs['tree'].xpath('./sense')])

    def _add_sense(self, sense):
        if 'name' in sense and isinstance(sense['name'], str) and len(sense['name']) > 0:
            self.sense2senseids[sense['name']].add(sense['id'])
        if 'lemma' in sense and isinstance(sense['lemma'], str) and len(sense['lemma']) > 0 :
            self.sense2senseids[sense['lemma']].add(sense['id'])
        # if 'main_word' in sense and isinstance(sense['main_word'], str) and len(sense['main_word']) > 0 :
        #     self.sense2senseids[sense['main_word']].add(sense['id'])
        self.senseid2synsetid[sense['id']] = sense['synset_id']
        self.synsetid2senseids[sense['synset_id']].add(sense['id'])


    def load_fom_dir(self, load_dir):
        files = [f
                 for f in os.listdir(load_dir)
                 if fs.isfile(fs.join(load_dir,f))
                 ]

        prefix2func = {
            'senses': self._load_senses,
            'synset_relations': self._load_synset_relations,
            'synsets': self._load_synsets,
            'composed_of': self._load_composed_of,
            #'derived_from': self._load_derived_from
        }

        for f in files:
            for prefix, func in prefix2func.items():
                if f.startswith(prefix):
                    func(filepath=fs.join(load_dir,f))
                    continue

        self.sense2senseids    = collections.defaultdict(set)
        self.senseid2synsetid  = collections.defaultdict()
        self.senseid2sensenum  = collections.defaultdict()

        self.synsetid2senseids = collections.defaultdict(set)

        for num, sense in enumerate(self.senses_list):
            self._add_sense(sense)
            self.senseid2sensenum[sense['id']] = num

        self.synsetid2synsetnum = collections.defaultdict()
        for num, synset in enumerate(self.synsets_list):
            for senseid in self.synsetid2senseids[synset['id']]:
                self.sense2senseids[synset['ruthes_name']].add(senseid)

            self.synsetid2synsetnum[synset['id']] = num

        # for composed_of in self.composed_of_list:
        #     root_sense = composed_of[0]
        #     for sense in composed_of[1]:
        #         self._add_sense(sense)
        #         self.sense2senseids[sense['name']].add(root_sense['id'])

        self.synset_relations = collections.defaultdict(set)
        for r in self.synset_relations_list:
            self.synset_relations[r['parent_id']].add( (r['child_id'], r['name']) )

    def make_synset_sentences(self,
                              ruthes_name=True,
                              definition=False,
                              senses_names=True,
                              senses_lemmas=True,
                              senses_main_word=False,
                              sep=' '
                              ):
        if not hasattr(self, 'synset_sentences'):
            self.synset_sentences = list()
            self._make_sent_ruthes_name = False
            self._make_sent_definition = False
            self._make_sent_senses_names = False
            self._make_sent_senses_lemmas = False
            self._make_sent_senses_main_word = False

        if (self._make_sent_ruthes_name == ruthes_name
            and self._make_sent_definition == definition
            and self._make_sent_senses_names == senses_names
            and self._make_sent_senses_lemmas == senses_lemmas
            and self._make_sent_senses_main_word == senses_main_word ):
            return

        self._make_sent_ruthes_name = ruthes_name
        self._make_sent_definition = definition
        self._make_sent_senses_names = senses_names
        self._make_sent_senses_lemmas = senses_lemmas
        self._make_sent_senses_main_word = senses_main_word

        symbol_counter = collections.Counter()

        for num, synset in enumerate(self.synsets_list):
            sentence = ''
            if ruthes_name:
                sentence += synset['ruthes_name'] + sep
            if definition:
                sentence += synset['definition'] + sep

            for senseid in self.synsetid2senseids[synset['id']]:
                sense = self.get_sense_by_id(senseid)
                if senses_names:
                    sentence += sense['name'] + sep
                if senses_lemmas:
                    sentence += sense['lemma'] + sep
                if senses_main_word:
                    sentence += sense['main_word'] + sep

            sentence = sentence.strip().lower()

            for symbol in sentence:
                symbol_counter[symbol] += 1

            if len(self.synset_sentences) == num:
                self.synset_sentences.append(sentence)
            elif len(self.synset_sentences) > num:
                self.synset_sentences[num] = sentence
            else:
                raise Exception ('What?!')

        logging.debug("symbol_counter %s ", symbol_counter)

    def get_synset_sentense(self, synsetid, *args, **kwargs):
        if not hasattr(self, 'synset_sentences'):
            self.make_synset_sentences(*args, **kwargs)

        return self.synset_sentences[self.synsetid2synsetnum[synsetid]]

    def get_synset_senses(self, synsetid, sep=','):
        return sep.join(self.get_synset_senses_list(synsetid))


    def get_synset_senses_list(self, synsetid):
        return sorted([self.get_sense_by_id(senseid)['name']
                       for senseid in self.synsetid2senseids[synsetid]]
                      )

    def get_senseids_by_sense(self, sense):
        sense = sense.upper()
        if sense in self.sense2senseids:
            return {senseid for senseid in self.sense2senseids[sense]}
        else:
            return set()

    def get_sense_by_id(self, senseid):
        return self.senses_list[self.senseid2sensenum[senseid]]


    def get_senses_by_sense(self, sense):
        return { self.senses_list[self.senseid2sensenum[senseid]]['name']
                 for senseid in self.get_senseids_by_sense(sense)
                }

    def get_synsetids_by_sense(self, sense):
        return { self.senseid2synsetid[senseid]
                 for senseid in self.get_senseids_by_sense(sense)
                }

    def get_synsets_by_sense(self, sense):
        return { self.synsets_list[self.synsetid2synsetnum[synsetid]]['ruthes_name']
                 for synsetid in self.get_synsetids_by_sense(sense)
                }

    def get_synset(self, synsetid):
        return self.synsets_list[self.synsetid2synsetnum[synsetid]]

    def get_name_by_id(self, synsetid):
        return self.get_synset(synsetid)['ruthes_name']

    def get_relation_by_id(self, synsetid, relation):
        assert (hasattr(self, 'synset_relations'))
        return [s_id  for s_id, rel in self.synset_relations[synsetid]
                      if rel==relation]

    def get_hypernyms_by_id(self, synsetid):
        return self.get_relation_by_id(synsetid, relation='hypernym')

    def get_hyponyms_by_id(self, synsetid):
        return self.get_relation_by_id(synsetid, relation='hyponym')

    def get_domains_by_id(self, synsetid):
        return self.get_relation_by_id(synsetid, relation='domain')




def main():
    load_dir = "/srv/data/develop/local/taxonomy/taxonomy-enrichment/data/ruwordnet"
    ruwordnet = RuWordNet(load_dir)

if __name__ == '__main__':
    main()
