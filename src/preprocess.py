import xml.etree.ElementTree as ET
import pandas as pd

def load_data(xml_path, limit=5000):
    rows = []

    for event, elem in ET.iterparse(xml_path):
        if elem.tag == "row" and elem.attrib.get("PostTypeId") == "1":

            body = elem.attrib.get("Body", "")
            title = elem.attrib.get("Title", "")
            tags = elem.attrib.get("Tags", "")

            rows.append({
                "Score": int(elem.attrib.get("Score", 0)),
                "ViewCount": int(elem.attrib.get("ViewCount", 0)),
                "AnswerCount": int(elem.attrib.get("AnswerCount", 0)),
                "CommentCount": int(elem.attrib.get("CommentCount", 0)),
                "FavoriteCount": int(elem.attrib.get("FavoriteCount", 0)),
                "title_length": len(title),
                "body_length": len(body),
                "word_count": len(body.split()),
                "num_tags": tags.count("|") // 2,
                "BodyText": body,
            })

        if len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = load_data("data/Posts.xml")
    print(df.head())
    print("\nShape:", df.shape)
