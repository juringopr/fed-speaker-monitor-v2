import streamlit as st


def render_references():
    st.subheader("References & Open-source Inspiration")

    st.markdown(
        """
This project was independently implemented for the **Fed Speaker Monitor**.

The following open-source repositories were reviewed as conceptual or
methodological references for news collection, financial-news processing,
RSS aggregation, Federal Reserve document collection, and text-analysis
workflows.

**The source code of this project is not a copy or reproduction of these
repositories.** The repositories below were used only as technical and
methodological references where applicable.

---

#### GNews

Google News collection and search methodology reference.

- **License:** MIT
- **GitHub:** https://github.com/ranahaani/GNews

#### FinNews

Financial-news collection and processing workflow reference.

- **License:** MIT
- **GitHub:** https://github.com/scaratozzolo/FinNews

#### MarketGPT

Financial text and LLM-analysis architecture reference.

- **License:** GPL
- **GitHub:** https://github.com/JHenzi/MarketGPT
- **No GPL source code is incorporated into this project.**

#### Finance News Aggregator

Multi-source financial-news aggregation methodology reference.

- **GitHub:** https://github.com/areed1192/finance-news-aggregator

#### Fed Statement Scraping

Federal Reserve document collection and scraping methodology reference.

- **GitHub:** https://github.com/vtasca/fed-statement-scraping
- **No source code from this repository is directly incorporated into this project.**

---

<small>
<b>Note:</b> References to the repositories above indicate conceptual or
methodological inspiration only. Fed Speaker Monitor uses its own
collection, normalization, deduplication, relevance-filtering,
segmentation, aggregation, and LLM-scoring implementation.

Third-party software licenses and the rights associated with collected
news content are separate matters.
</small>
""",
        unsafe_allow_html=True,
    )