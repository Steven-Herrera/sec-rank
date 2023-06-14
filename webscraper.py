from bs4 import BeautifulSoup
import requests

class sec_webscraper:
    def __init__(self, rule_URL):
        self.rule_URL = rule_URL

    # function to retrieve names and links for each comment in a rule
    def rule_comment_info_retriever(self):
        #rule_URL = "https://www.sec.gov/comments/s7-21-21/s72121.htm"
        rule_page = requests.get(self.rule_URL, timeout=15)
        rule_soup = BeautifulSoup(rule_page.content, "html.parser")

        remove_from_title = ['SEC.gov', " | "]
        rule_title = rule_soup.title.string
        for string in remove_from_title:
            if string in rule_title:
                rule_title = rule_title.replace(string, "")
        rule_title = rule_title.strip()

        commenter_links = []
        commenter_names = []
        commenter_dates = []
        letter_form_links, letter_form_types = self.find_letter_forms_v1(rule_soup)
        comment_dates, comment_names, comment_links = self.find_dates_names(rule_soup)
        
        if letter_form_types != None:
            commenter_names.extend(letter_form_types)
            
        commenter_names.extend(comment_names)
        
        if letter_form_links != None:
            commenter_links.extend(letter_form_links)
            
        commenter_links.extend(comment_links)
        if letter_form_links != None:
            none_dates = [None for i in range(len(letter_form_links))]
            commenter_dates.extend(none_dates)
        
        commenter_dates.extend(comment_dates)
        commenter_links = self.link_creator(commenter_links)
        
        return commenter_names, commenter_links, commenter_dates

    def link_creator(self, sublink_list):
        link_list = []
        sec_webpage = 'https://www.sec.gov'
        for sublink in sublink_list:
            link = sec_webpage + sublink
            link_list.append(link)
        return link_list

    # This version gets the letter form count
    def find_letter_forms_v1(self, comment_soup):
        result_set = comment_soup.find_all('td')
        for x in result_set:
            if "Letter Type" in str(x):
                no_html_comments = self.remove_html_comments(x)
                letter_forms = self.multi_href_handler(no_html_comments)
        letterForm_letters = []
        try:
            for x in no_html_comments:
                split = x.split('">')
                for s in split:
                    letter_and_count = s.split('<')[0]
                    if ":" in letter_and_count:
                        letterForm_letters.append(letter_and_count)
        except (AttributeError, NameError, TypeError):
            letter_forms = None
            letterForm_letters = None
            
        return letter_forms, letterForm_letters # pylint: disable=used-before-assignment

    def find_dates_names(self, comment_soup):
        i=0
        dates = []
        names = []
        links = []
        result_set = comment_soup.find_all('tr')
        for x in result_set:
            a = str(x.a)
            s = str(x.td)
            # first instance of meetings is a reference to the bottom of the webpage
            if "meetings" in s:
                i+=1
            # obtain dates
            elif "nobreak" in s:
                d = s.split('>')[1]
                d = d.split('<')[0]
                dates.append(d)
            # in the Cybersecurity rule there is a commenter: 'Magda Liliana Zambrano Barragan' who's href does
            # not contain the a `/comments/file_no` handle. I deal with this one off mistake in the html in
            # if statement. Her href is `s70922-291299.htm`
            if (("comments" in a) or ("s70922-291299.htm" in a)) and ("type" not in a) and ("</h3>" not in a):                
                names.append(str(x.a.string))
            # `meetings` is sometimes in the tr tag and these are for memorandums
            # since Magda's href doesn't contain `/comments/file_no` we have to correct it
            if ('<td><a href="' in str(x) or ("s70922-291299.htm" in str(x))) and ("meetings" not in str(x)):
                if ("s70922-291299.htm" in str(x)):
                    sublink = '/comments/s7-09-22/s70922-291299.htm'
                    links.append(sublink)
                else:
                    something = str(x).split('<td><a href="')[1]
                    something = something.split('">')[0]
                    links.append(something)
            # second instance of 'meetings' list the memorandums
            if ("meetings" in s) and i>1:
                break
        return dates, names, links

    def remove_html_comments(self, string):
        temp_strings_list = str(string).split('-->')
        temp_list = []
        for x in temp_strings_list:
            temp_s = x.split('<!--')[0]
            temp_list.append(temp_s)
        return temp_list

    def multi_href_handler(self, href_tags):
        clean_split_hrefs = []
        multiple_hrefs = str(href_tags)
        split_hrefs = multiple_hrefs.split("<a href=")
        for string in split_hrefs:
            if "/comments/" in string:
                href_string = string
                clean_href_string = href_string.split('">')[0][1:]
                clean_split_hrefs.append(clean_href_string)
        return clean_split_hrefs