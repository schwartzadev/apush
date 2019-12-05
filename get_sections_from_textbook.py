import requests
from bs4 import BeautifulSoup


def get_chapter_sections(chapter_url):
	data = requests.get(chapter_url)
	soup = BeautifulSoup(data.content, features="lxml")
	entry_content = soup.find('div', {'class': 'entry-content'}) # get the main section

	# get the p elements and the section header elements
	content = [w for w in entry_content if (w.name == 'h1' and w.get('style') == 'text-align: center;') or (w.name == 'p')]


	# organize section information into a data structure
	textbook_data = []
	current_header = None
	current_tags = []

	for element in content:
		if element.name == 'h1': # this is a header!
			if current_header is not None:
				textbook_data.append({'header': current_header.text, 'content': [c.text for c in current_tags]})
			else:
				textbook_data.append({'header': None, 'content': current_tags})
			current_header = element # reset
			current_tags = [] # reset
		else: # this is a p element -- add it to the list
			current_tags.append(element)

	textbook_data = textbook_data[1:] # remove first (blank) content section

	return textbook_data


if __name__ == '__main__':
	chapter_eight = get_chapter_sections('http://www.americanyawp.com/text/08-the-market-revolution/')
	print(len(chapter_eight), 'sections found!')
	# print(chapter_eight)
