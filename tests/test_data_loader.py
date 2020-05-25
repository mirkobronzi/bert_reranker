import json

from bert_reranker.data.data_loader import get_passages_by_source

SIMPLE_JSON = """
{
      "examples": [],
      "passages": [
            {
                  "passage_id": 0,
                  "source": "my_source",
                  "reference_type": "faq",
                  "reference": {
                        "page_title": "",
                        "section_headers": [
                              "my question"
                        ],
                        "section_content": "",
                        "selected_span": null
                  }
            }
      ]
}
"""


def test_get_passages_by_source__simple():
    source2passages, pid2passage, pid2index = get_passages_by_source(
        json.loads(SIMPLE_JSON)
    )
    assert pid2index == {0: 0}


ID_AND_OOD_JSON = """
{
      "examples": [],
      "passages": [
            {
                  "passage_id": 0,
                  "source": "my_source",
                  "reference_type": "faq",
                  "reference": {
                        "page_title": "",
                        "section_headers": [
                              "my question"
                        ],
                        "section_content": "",
                        "selected_span": null
                  }
            },
            {
                  "passage_id": 12,
                  "source": "my_source",
                  "reference_type": "non_faq",
                  "reference": {
                        "page_title": "",
                        "section_headers": [],
                        "section_content": "",
                        "selected_span": null
                  }
            }
      ]
}
"""


def test_get_passages_by_source__id_and_ood():
    source2passages, pid2passage, pid2index = get_passages_by_source(
        json.loads(ID_AND_OOD_JSON)
    )
    assert pid2index == {0: 0, 12: -1}
