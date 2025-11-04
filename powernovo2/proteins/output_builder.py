from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from pandas import concat, DataFrame


class TableMaker(object):
    def get_protein_table(self, pn):
        # source the data as a list
        df = self._get_edge_list(pn)

        agg_dict = self._get_agg_dict(pn)
        protein_df = df.groupby("protein_id").agg(agg_dict)

        protein_df["total_peptides"] = df.groupby("protein_id").size()


        if pn.get_node_attribute_dict("unique"):
            protein_df["non_unique"] = protein_df.total_peptides - protein_df.unique
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        for col in ["razor", "non_unique", "unique"]:
            if pn.get_node_attribute_dict(col):
                protein_df[col] = protein_df[col].astype("int32")

        if pn.get_node_attribute_dict("non_unique"):
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        if pn.get_node_attribute_dict("non_unique"):
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        if pn.get_node_attribute_dict("ids"):
            protein_df['ids'] = protein_df.name.astype("str")

        if pn.get_node_attribute_dict("name"):
            protein_df['name'] = protein_df.name.astype("str")

        # sort sequence modified
        protein_df.sequence_modified = protein_df.sequence_modified.apply(lambda x: sorted(x))

        protein_df = protein_df.reset_index()  # otherwise protein id isn't a column

        # if solved, add new scores:
        dict_score = pn.get_node_attribute_dict("score")
        if dict_score:
            protein_df["score"] = protein_df.protein_id.apply(
                lambda x: dict_score[x])

        # if solved, add subset proteins:
        dict_subset = pn.get_node_attribute_dict("major")
        if dict_subset:
            protein_df["Group"] = protein_df.protein_id.apply(
                lambda x: dict_subset[x])

        if pn.get_node_attribute_dict("indistinguishable"):
            protein_df = self.add_indistinguishable_col(pn, protein_df)
            protein_df.indistinguishable = protein_df.indistinguishable.apply(lambda x: sorted(x))

        if pn.get_node_attribute_dict("major"):
            protein_df = self.add_subset_col(pn, protein_df)
            protein_df.subset = protein_df.subset.apply(lambda x: sorted(x))

        cols = ["protein_id",
                "name",
                "unique",
                "non_unique",
                "razor",
                "total_peptides",
                "score",
                "ids",
                "Group",
                "indistinguishable",
                "subset",
                "sequence_modified"]

        new_cols = []
        for col in cols:
            if col in protein_df.columns:
                new_cols.append(col)

        protein_df = protein_df.loc[:, new_cols]

        if "score" in protein_df.columns:
            return protein_df.sort_values("score", ascending=False)
        else:
            return protein_df

    def get_protein_tables(self, pns):

        p = Pool(cpu_count())
        protein_tables = p.map(self.get_protein_table, pns)

        return protein_tables

    def get_system_protein_table(self, pns):
        protein_table = concat(self.get_protein_tables(pns))
        protein_table = self.emulate_percolator_formatting(protein_table)
        return protein_table


    @staticmethod
    def get_peptide_table(pn):
        def label_score(protein, score_dict):
            return score_dict[protein]

        score_dict = pn.get_node_attribute_dict("score")
        df = TableMaker()._get_edge_list(pn)

        df = TableMaker()._get_edge_list(pn)

        # агрегируем по пептиду, берем макс peptide_score среди связей пептид↔белок
        agg = {
            "ids": lambda x: str(list(x)[0]) if len(x) else None,
            "scan_id": "first",
            "denovo_sequence": "first",
            "denovo_score": "first",
            "unique": "min",
            "razor": "min",
            "unique_evidence": "max",
            "major": "min",
            "score": "min",
            "protein_name": "min",
            "protein_id": list,
            "peptide_score": "max",  # NEW
        }
        if "pepide_ppm_diff" in df.columns:
            agg["pepide_ppm_diff"] = "min"

        df = df.groupby("sequence_modified").aggregate(agg).reset_index()

        df["scan_id"] = df["scan_id"].fillna(df["ids"]).astype(str)
        df = df.rename(columns={
            "sequence_modified": "peptide",
            "protein_name": "major_name",
            "protein_id": "all_proteins",
            "score": "identity"
        })
        for col in ["unique_evidence", "ids"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df_prot_score = df[["all_proteins", "peptide_score"]].explode("all_proteins")
        df_prot_score = df_prot_score.dropna(subset=["all_proteins"])
        prot_max = df_prot_score.groupby("all_proteins")["peptide_score"].max().rename("protein_score")
        def peptide_protein_score_max(row):
            proteins = row["all_proteins"] if isinstance(row["all_proteins"], list) else []
            vals = [prot_max.get(p, np.nan) for p in proteins]
            vals = [v for v in vals if not pd.isna(v)]
            return max(vals) if vals else np.nan
        df["protein_score"] = df.apply(peptide_protein_score_max, axis=1)

        cols = [
            "peptide",
            "denovo_sequence",
            "scan_id",
            "unique",
            "razor",
            "identity",
            "pepide_ppm_diff",
            "denovo_score",
            "peptide_score",   # есть в выходе
            "major",
            "major_name",
            "protein_score",
            "all_proteins",
        ]
        existing = [c for c in cols if c in df.columns]
        df = df.loc[:, existing]

        sort_cols = [c for c in ["protein_score", "peptide_score"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        return df

    def get_peptide_tables(self, pns):

        p = Pool(cpu_count())
        peptide_tables = p.map(self.get_peptide_table, pns)

        return peptide_tables

    def get_system_peptide_table(self, pns):
        peptide_table = concat(self.get_peptide_tables(pns))
        return peptide_table

    @staticmethod
    def _get_edge_list(pn):
        rows = []
        for u, v, d in pn.network.edges(data=True):
            node_1_data = pn.network.nodes[u]
            node_2_data = pn.network.nodes[v]
            row = dict()
            if node_1_data.get('is_protein', 0) == 1:

                row['protein_id'] = u
                row['sequence_modified'] = v
            else:

                row['protein_id'] = v
                row['sequence_modified'] = u

            row.update({k: val for k, val in node_1_data.items() if k != 'is_protein'})
            row.update({k: val for k, val in node_2_data.items() if k != 'is_protein'})

            row.update(d)
            rows.append(row)

        df = DataFrame(rows)

        if 'is_protein' in df.columns:
            df = df.drop(columns=['is_protein'])
        return df


    @staticmethod
    def _flip_dict(old_dict):
        new_dict = {}
        for key, value in old_dict.items():
            if value in new_dict:
                new_dict[value].append(key)
            else:
                new_dict[value] = [key]
        return new_dict

    @staticmethod
    def add_indistinguishable_col(pn, table):

        indistinguishable_dict = pn.get_node_attribute_dict("indistinguishable")
        new_col = []
        for _, row in table.iterrows():
            new_col.append(indistinguishable_dict[row["protein_id"]])

        table["indistinguishable"] = new_col

        return table

    def add_subset_col(self, pn, table):

        subset_dict = pn.get_node_attribute_dict("major")
        subset_dict = self._flip_dict(subset_dict)
        new_col = []
        for _, row in table.iterrows():
            if row["protein_id"] == row["Group"]:  # and row["protein_id"] in subset_dict.keys():
                new_col.append(subset_dict[row["protein_id"]])
            else:
                new_col.append([])

        table["subset"] = new_col

        return table

    @staticmethod
    def _get_agg_dict(pn):
        agg_dict = {}

        agg_dict.update({"ids": list})

        if pn.get_node_attribute_dict("razor"):
            agg_dict.update({"razor": 'sum'})
        if pn.get_node_attribute_dict("unique"):
            agg_dict.update({"unique": 'sum'})
        if pn.get_node_attribute_dict("score"):
            agg_dict.update({"score": 'sum'})

        # always add sequence_modified
        agg_dict.update({"sequence_modified": list})
        agg_dict.update({"name": "first"})

        return agg_dict

    @staticmethod
    def emulate_percolator_formatting(protein_table):
        col_dict = {"protein_id": "ProteinId", "sequence_modified": "peptideIds"}
        protein_table = protein_table.rename(columns=col_dict)
        protein_table = protein_table.sort_values("ProteinId")
        protein_table["peptideIds"] = protein_table.peptideIds.apply(lambda x: " ".join(x))
        labels, _ = pd.factorize(protein_table.Group)
        protein_table["ProteinGroupId"] = labels

        return protein_table
