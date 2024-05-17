import spacy
from spacy import displacy


class TreeLocate():
    def __init__(self):
        self.nlp=spacy.load("en_core_web_sm")

    def render_tree(doc):
        displacy.render(doc,style="dep",jupyter=True,options={'distance':90})

    def get_common_words(self,list1, list2):
        common_words = [word for word in list1 if word in list2]
        return common_words
    
    def get_children(self,chunk):
        return [r.text for r in chunk.root.children]
    
    # def get_children(self, chunk):
    #     texts = []
    #     texts.append(chunk.text)
    #     for child in chunk.children:
    #         texts.extend(self.get_children(child))
    #     return texts

    def get_one_chunk(self,chunks,name):
        for i,chunk in enumerate(chunks):
            if chunk.root.text==name:
                return i
        else :
            return -1

    def get_tree(self,docs):
        root_entity=[]
        vice_entity=[]
        for nps in docs.noun_chunks:
            print(nps.text, nps.root.dep_, nps.root.head.text)
            if nps.root.dep_=="ROOT":
                root_entity.append(nps)
            elif nps.root.dep_=="conj":
                if nps.root.head.dep_=="ROOT":
                    root_entity.append(nps)
                else:
                    vice_entity.append(nps)
            else:
                vice_entity.append(nps)
        return root_entity,vice_entity
    
    def get_tree_node(self,docs):
        entity=[]
        for nps in docs.noun_chunks:
            entity.append(nps)
        return entity
    
    def locate_node(self,origin_caption,input_prompt):
        origin_doc=self.nlp(origin_caption)
        root_o=self.get_tree_node(origin_doc)

        input_doc=self.nlp(input_prompt)
        root_update=self.get_tree_node (input_doc)

        decrease_list=[]
        root_old_nn=[n.root.text for n in root_o]
        root_update_nn=[n.root.text for n in root_update]

        common_entity=self.get_common_words(root_old_nn,root_update_nn)
        if len(common_entity)==0:
            for r in root_o:
                decrease_list.append(r.text)
        else:
            for old in root_o:
                if old.root.text in root_update_nn:
                    root_index=self.get_one_chunk(root_update,old.root.text)
                    common_chunk=root_update[root_index]
                    # 更新属性
                    old_children=self.get_children(old)
                    update_children=self.get_children(common_chunk)
                    decrease_list+=list(set(old_children)- set(update_children))
                    # decrease_list+=[x for x in old_children if x not in update_children]

                else:
                    decrease_list.append(old.text)
        
        # 创建doc对象
        doc = self.nlp(' '.join(decrease_list))

        # 删除停用词和标点
        filtered_text = [token.text for token in doc if (not token.is_stop) and (not token.is_punct)]
        return ' '.join(filtered_text)
    
    def locate(self,origin_caption,input_prompt):
        origin_doc=self.nlp(origin_caption)
        root_o,vice_o=self.get_tree(origin_doc)

        input_doc=self.nlp(input_prompt)
        root_update,vice_update=self.get_tree(input_doc)

        decrease_list=[]
        root_old_nn=[n.root.text for n in root_o]
        root_update_nn=[n.root.text for n in root_update]

        common_entity=self.get_common_words(root_old_nn,root_update_nn)
        if len(common_entity)==0:
            for r in root_o:
                decrease_list.append(r.text)
        else:
            for old in root_o:
                if old.root.text in root_update_nn:
                    root_index=self.get_one_chunk(root_update,old.root.text)
                    common_chunk=root_update[root_index]
                    old_children=self.get_children(old)
                    update_children=self.get_children(common_chunk)
                    decrease_list+=list(set(old_children) - set(update_children))

        # 创建doc对象
        doc = self.nlp(' '.join(decrease_list))

        # 删除停用词
        filtered_text = [token.text for token in doc if not token.is_stop]
        return decrease_list

# nlp=TreeLocate()
# result=nlp.locate_node("A seat under a mirror onboard a train, next to a cluttered counter.","A cat lounges on a seat under a mirror onboard a train, next to a cluttered counter.")
# print(result)