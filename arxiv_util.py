import arxiv
from tqdm import tqdm
from datetime import datetime
from datetime import datetime, timezone
import pandas as pd

def query_with_keywords(query, client) -> tuple:
    """
    Query the arXiv API for research papers based on a specific query and filter results by selected categories.
    
    Args:
        query (str): The search query to be used for fetching research papers from arXiv.
    
    Returns:
        tuple: A tuple containing 5 lists - terms, titles, and abstracts ,url and id of the filtered research papers.
        
            terms (list): A list of lists, where each inner list contains the categories associated with a research paper.
            titles (list): A list of titles of the research papers.
            abstracts (list): A list of abstracts (summaries) of the research papers.
            urls (list): A list of URLs for the papers' detail page on the arXiv website.
            ids (list):unique ids of the paper
    """
    
    # Create a search object with the query and sorting parameters.
    search = arxiv.Search(
        query=query,
        max_results=6000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    
    # Initialize empty lists for terms, titles, abstracts, and urls.
    terms = []
    titles = []
    abstracts = []
    urls = []
    ids = []
    authors = []
    publisheds = []
    journals = []
    
    # For each result in the search...
    for res in tqdm(client.results(search), desc=query):
        #print(res)
        # Check if the primary category of the result is in the specified list.
        if res.primary_category in ["cs.CV", "stat.ML", "cs.LG", "cs.AI" ,"cs.CL"]:
            # If it is, append the result's categories, title, summary, and url to their respective lists.
            terms.append(res.categories)
            titles.append(res.title)
            abstracts.append(res.summary)
            urls.append(res.entry_id)
            ids.append(res.entry_id.split('/')[-1])
            authors.append(res.authors)
            publisheds.append(res.published)
            journals.append(res.journal_ref)

    # Return the eight lists.
    return terms, titles, abstracts, urls , ids, authors, publisheds, journals


def query_with_keywords_and_date_range(query, client,start_date, end_date) -> tuple:
    """
    Query the arXiv API for research papers based on a specific query and filter results by selected categories and date range.
    
    Args:
        query (str): The search query to be used for fetching research papers from arXiv.
        start_date (str): The start date for the date range filter in YYYY-MM-DD format.
        end_date (str): The end date for the date range filter in YYYY-MM-DD format.
    
    Returns:
        tuple: A tuple containing lists - terms, titles, abstracts, urls, ids, authors, publisheds, and journals of the filtered research papers.
    """
    
    # Create a search object with the query and sorting parameters.
    search = arxiv.Search(
        query=query,
        max_results=6000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )

    # Initialize empty lists for terms, titles, abstracts, urls, ids, authors, publisheds, and journals.
    terms = []
    titles = []
    abstracts = []
    urls = []
    ids = []
    authors = []
    publisheds = []
    journals = []
    
    # Convert start_date and end_date to datetime objects.
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = end_date.replace(tzinfo=timezone.utc)

    # For each result in the search...
    for res in tqdm(client.results(search), desc=query):
        # Convert the published date of the paper to a datetime object.
        # published_date = datetime.strptime(res.published, "%Y-%m-%dT%H:%M:%SZ")
        published_date = res.published

        # Check if the published date is within the specified date range.
        if start_date <= published_date <= end_date:
            # Check if the primary category of the result is in the specified list.
            if res.primary_category in ["cs.CV", "stat.ML", "cs.LG", "cs.AI", "cs.CL"]:
                # Append the result's categories, title, summary, and url to their respective lists.
                terms.append(res.categories)
                titles.append(res.title)
                abstracts.append(res.summary)
                urls.append(res.entry_id)
                ids.append(res.entry_id.split('/')[-1])
                authors.append(res.authors)
                publisheds.append(res.published)
                journals.append(res.journal_ref)

    # Return the eight lists.
    return terms, titles, abstracts, urls, ids, authors, publisheds, journals




def fetch_arxiv(client, query_keywords,start_date, end_date):

    # list of schema
    all_titles = []
    all_abstracts = []
    all_terms = []
    all_urls = []
    all_ids = []
    all_authors = []
    all_publisheds = []
    all_journals = []

    terms, titles, abstracts, urls , ids, authors, publisheds, journals = query_with_keywords_and_date_range(query_keywords, client, start_date, end_date)
    all_titles.extend(titles)
    all_abstracts.extend(abstracts)
    all_terms.extend(terms)
    all_urls.extend(urls)
    all_ids.extend(ids)
    all_authors.extend(authors)
    all_publisheds.extend(publisheds)
    all_journals.extend(journals)


    arxiv_data = pd.DataFrame({
        'titles': all_titles,
        'abstracts': all_abstracts,
        'terms': all_terms,
        'urls': all_urls,
        'ids': all_ids,
        'authors':all_authors,
        'publishes': all_publisheds,
        'journals': all_journals,
    })

    return arxiv_data

