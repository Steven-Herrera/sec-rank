# function to retrieve names and links for each comment in a rule
def rule_comment_info_retriever(rule_URL):
    #rule_URL = "https://www.sec.gov/comments/s7-21-21/s72121.htm"
    rule_page = requests.get(rule_URL)
    rule_soup = BeautifulSoup(rule_page.content, "html.parser")

    remove_from_title = ['SEC.gov', " | "]
    rule_title = rule_soup.title.string
    for string in remove_from_title:
        if string in rule_title:
            rule_title = rule_title.replace(string, "")
    rule_title = rule_title.strip()

    link_list = rule_soup.find_all('a')
    split_url = rule_URL[:-4].split("/")[-2:]
    prefix_0 = "/".join(split_url)
    prefix = """<a href="/comments/""" + prefix_0 + "-"
    URL_prefix = "https://www.sec.gov"
    commenter_links = []
    commenter_names = []
    commenter_dates = []
    letter_form_links, letter_form_types = find_letter_forms_v1(rule_soup)
    comment_dates, comment_names, comment_links = find_dates_names(rule_soup)
    
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
    commenter_links = link_creator(commenter_links)
    
    return commenter_names, commenter_links, commenter_dates


# In[88]:


def link_creator(sublink_list):
    link_list = []
    sec_webpage = 'https://www.sec.gov'
    for sublink in sublink_list:
        link = sec_webpage + sublink
        link_list.append(link)
    return link_list

def multi_href_handler(href_tags):
    clean_split_hrefs = []
    multiple_hrefs = str(href_tags)
    split_hrefs = multiple_hrefs.split("<a href=")
    for string in split_hrefs:
        if "/comments/" in string:
            href_string = string
            #print(f"""href split: {href_string.split('">')}""")
            clean_href_string = href_string.split('">')[0][1:]
            clean_split_hrefs.append(clean_href_string)
            #print(clean_split_hrefs)
    return clean_split_hrefs

def remove_html_comments(string):
    temp_strings_list = str(string).split('-->')
    temp_list = []
    for x in temp_strings_list:
        temp_s = x.split('<!--')[0]
        temp_list.append(temp_s)
    return temp_list

# This version doesn't get the letter form count
def find_letter_forms(comment_soup):
    result_set = comment_soup.find_all('td')
    temp_list = []
    for x in result_set:
        if "Letter Type" in str(x):
            no_html_comments = remove_html_comments(x)
            letter_forms = multi_href_handler(no_html_comments)
    letterForm_letters = []
    try:
        for link in letter_forms:
            temp_string = link.split('-')[-1]
            letter_type = temp_string.split('.')[0]
            letterForm_letters.append(letter_type)
    except:
        letter_forms = None
        letterForm_letters = None
        
    return letter_forms, letterForm_letters

# This version gets the letter form count
def find_letter_forms_v1(comment_soup):
    result_set = comment_soup.find_all('td')
    temp_list = []
    for x in result_set:
        if "Letter Type" in str(x):
            no_html_comments = remove_html_comments(x)
            letter_forms = multi_href_handler(no_html_comments)
    letterForm_letters = []
    try:
        for x in no_html_comments:
            split = x.split('">')
            for s in split:
                letter_and_count = s.split('<')[0]
                if ":" in letter_and_count:
                    letterForm_letters.append(letter_and_count)
    except:
        letter_forms = None
        letterForm_letters = None
        
    return letter_forms, letterForm_letters

def find_dates_names(comment_soup):
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

def make_hyperlink(link_list, name_list):
    hyper_strings = []
    for i in range(len(link_list)):
        url = link_list[i]
        name = name_list[i]
        hyper_string = f'=HYPERLINK("{url}", "{name}")'
        hyper_strings.append(hyper_string)
    return hyper_strings

def rule_nickname_lookup(rule_header):
    nickname_dict = {'Comments for The Enhancement and Standardization of Climate-Related Disclosures for Investors':'Climate',
 'Comments on the Cybersecurity Risk Management, Strategy, Governance, and Incident Disclosure':'Cyber',
 'Comments on Modernization of Beneficial Ownership Reporting':'Beneficial Ownership',
 'Comments on Share Repurchase Disclosure Modernization':'Buybacks',
 'Comments on Removal of References to Credit Ratings From Regulation M':'Removal of References to Credit Ratings From Regulation M',
 'Comments on Substantial Implementation, Duplication, and Resubmission of Shareholder Proposals Under Exchange Act Rule 14a-8':'Shareholder Proposals Rule 14a-8',
 'Comments on Special Purpose Acquisition Companies, Shell Companies, and Projections':'SPACs'}
    try:
        nickname = nickname_dict[rule_header]
        return nickname
    except KeyError:
        nickname = rule_header.strip("Comments for ")
        nickname = nickname.strip("Comments on ")
        #print(f"No Ni. Returning: {nickname}")
        return nickname
    
def PDF_Outliner_v3(pdf_list, rule_path, chunk_num, latest_folder):
    
    outline_dir = os.path.join(latest_folder, f"Outlined_PDFs")
    chunk_j = os.path.join(outline_dir, f"Outline_Chunk_{chunk_num}")
    if not os.path.exists(chunk_j):
        os.mkdir(chunk_j)

    bookmark_indices_j = []
    bookmark_names_j = []
    
    for k in range(len(pdf_list)):
        commenter_name = pdf_list[k].split("/")[-1].split("_")[1]   
        commenter_filename = pdf_list[k].split("/")[-1]
        pdf_newpath = os.path.join(chunk_j, commenter_filename)
        writer = PdfWriter()
        reader = PdfReader(pdf_list[k])
        writer.add_page(reader.pages[0])
        writer.add_outline_item(commenter_name, 0, parent=None)
        
        bookmark_names_j.append(commenter_name)
        bookmark_indices_j.append(len(reader.pages))
        
        for p in range(1, len(reader.pages)):
            writer.add_page(reader.pages[p])
        with open(pdf_newpath, 'wb') as fp:
            writer.write(fp)

    # merge newly bookmarked PDF files
    outlined_pdfs = sorted(os.listdir(chunk_j))
    merge = PdfMerger(strict=False)
    for k in range(len(outlined_pdfs)):
        outline_pdf_path = os.path.join(chunk_j, outlined_pdfs[k])
        merge.append(outline_pdf_path, import_outline = True)
    merged_filename = f"{latest_folder.split('/')[-1]}_Chunk_{chunk_num}.pdf"
    merged_filepath = os.path.join(outline_dir, merged_filename)
    merge.write(merged_filepath)
    merge.close()
    
    return bookmark_indices_j, bookmark_names_j, merged_filepath
    
def climate_outliner_v1(rule_path):
    # most recent folder
    latest_folder = find_latest_folder(rule_path)
    # create a new folder for PDFs with bookmarks
    outline_dir = 'Outlined_PDFs'
    outline_path = os.path.join(latest_folder, outline_dir)
    if not os.path.exists(outline_path):
        os.mkdir(outline_path)
    
    # get PDFs from latest folder
    latest_temp_pdf_dir = os.path.join(latest_folder, "Temp_PDFs")
    latest_pdfs = os.listdir(latest_temp_pdf_dir)
    latest_pdfs_paths = sorted([os.path.join(latest_temp_pdf_dir, pdf) for pdf in latest_pdfs])
    
    num_files = len(latest_pdfs_paths)
    OPEN_FILE_LIMIT = 1000
    num_chunks = int(np.ceil(num_files / OPEN_FILE_LIMIT))
    chunk_filepaths = []
    
    bookmark_indices = []
    bookmark_names = []
    
    for i in range(num_chunks):
        if i < (num_chunks - 1):
            pdf_chunk_list = latest_pdfs_paths[i*OPEN_FILE_LIMIT:(i+1)*OPEN_FILE_LIMIT]
            bm_i, bm_n, chunk_fp = PDF_Outliner_v3(pdf_chunk_list, rule_path, i, latest_folder)
            bookmark_indices.append(bm_i)
            bookmark_names.append(bm_n)
            chunk_filepaths.append(chunk_fp)
        else:
            pdf_chunk_list = latest_pdfs_paths[i*OPEN_FILE_LIMIT:]
            bm_i, bm_n, chunk_fp = PDF_Outliner_v3(pdf_chunk_list, rule_path, i, latest_folder)
            bookmark_indices.append(bm_i)
            bookmark_names.append(bm_n)
            chunk_filepaths.append(chunk_fp)
            
    chunk_readers = []
    chunk_outlines = []
    for i in range(len(chunk_filepaths)):
        reader = PdfReader(chunk_filepaths[i])
        outlines = reader.outline

        chunk_readers.append(reader)
        chunk_outlines.append(outlines)
        
    writer = PdfWriter()
    page_count = 0
    for i in range(len(chunk_readers)):
        num_pages = len(chunk_readers[i].pages)
        for p in range(num_pages):
            writer.add_page(chunk_readers[i].pages[p])

        num_bookmarks = len(chunk_readers[i].outline)
        for b in range(num_bookmarks):
            bookmark = chunk_readers[i].outline[b]
            bm_page = bookmark['/Page']
            bm_name = bookmark['/Title']
            writer.add_outline_item(bm_name, page_count+bm_page, parent=None)

        page_count += num_pages
        
    binder_name = latest_folder.split("/")[-1]
    pdf_newpath = os.path.join(latest_folder, f"{binder_name}.pdf")
    with open(pdf_newpath, 'wb') as fp:
        writer.write(fp)

    shutil.rmtree(latest_temp_pdf_dir)

    for i in range(len(chunk_filepaths)):
        os.remove(chunk_filepaths[i])
        chunk_path = os.path.join(outline_path, f"Outline_Chunk_{i}")
        outline_pdfs = os.listdir(chunk_path)
        outline_paths = [os.path.join(chunk_path, pdf) for pdf in outline_pdfs]
        new_paths = [os.path.join(outline_path, pdf) for pdf in outline_pdfs]
        for j in range(len(outline_paths)):
            shutil.move(outline_paths[j], new_paths[j])
        os.rmdir(chunk_path)
        
# each thread records the links in its own txt file
def record_link(link, path_to_link_keeper):
    # a txt file to keep track of links previously scraped in the past
    #path_to_link_keeper = os.path.join(rule_path, "Downloaded_Links_i.txt")
    if os.path.exists(path_to_link_keeper):
        # write to record keeper if it exists
        with open(path_to_link_keeper, 'r') as record_keeper:
            recorded_links = record_keeper.read().split()
        if link not in recorded_links:
            link_string = str(link) + '\n'
            with open(path_to_link_keeper, 'a') as record_keeper:
                record_keeper.write(link_string)
    else:
        # if it doesn't exist create and write
        with open(path_to_link_keeper, 'w') as record_keeper:
            link_string = str(link) + '\n'
            record_keeper.write(link_string)

            
def undownloaded_links_names(links, names, dates, path_to_link_keeper):
    undownloaded_links = []
    undownloaded_names = []
    undownloaded_dates = []
    if os.path.exists(path_to_link_keeper):
        with open(path_to_link_keeper, 'r') as record_keeper:
            recorded_links = record_keeper.read().split()
        for i in range(len(links)):
            link = links[i]
            name = names[i]
            date = dates[i]
            if link in recorded_links:
                pass
            else:
                undownloaded_links.append(link)
                undownloaded_names.append(name)
                undownloaded_dates.append(date)
    else:
        with open(path_to_link_keeper, 'w') as record_keeper:
            pass
        for i in range(len(links)):
            undownloaded_links.append(links[i])
            undownloaded_names.append(names[i])
            undownloaded_dates.append(dates[i])
            
    return undownloaded_links, undownloaded_names, undownloaded_dates

# do pdf downloads
def download_webpage_v2(undownloaded_links, undownloaded_names, undownloaded_dates,
                                this_wk_binder_path, this_wk_binder_name, rule_path):
    # someone could write more than 1 comment
    links = undownloaded_links
    names = undownloaded_names
    #this_wk_binder_name = params['BINDER']
    directory = this_wk_binder_path
    dates = undownloaded_dates
    path_to_link_keeper = os.path.join(rule_path, "Download_Links.txt")
    #thread_number = params['NUM']
    desc = f"{this_wk_binder_name[:-16]}"
    # this is just to store each individual PDF until we can merge them all together
    temporary_pdf_directory = os.path.join(directory, "Temp_PDFs")
    if not os.path.exists(temporary_pdf_directory):
        os.mkdir(temporary_pdf_directory)

    for i in tqdm.tqdm(range(len(links)), desc=desc):
        # the weblink could be htm, txt, or pdf
        link_type = links[i][-3:]
        # Naming the pdf file after the person that wrote it, submission date, and link
        pdf_filename = pdf_filenamer(names[i], links[i], dates[i])

        pdf_filepath = os.path.join(temporary_pdf_directory, pdf_filename)
        try:
            # download comment and write link to link record keeper
            if (link_type == "htm"):
                converter.convert(links[i], pdf_filepath)
                record_link(links[i], path_to_link_keeper)
            elif (link_type == "pdf"):
                comment_page = requests.get(links[i])
                with open(pdf_filepath, "wb") as pdf:
                    pdf.write(comment_page.content)
                record_link(links[i], path_to_link_keeper)
            elif (link_type == "txt"):
                comment_page = requests.get(links[i])
                comment_soup = BeautifulSoup(comment_page.content, "html.parser")
                txt2pdf(pdf_filepath,md_content=str(comment_soup))
                record_link(links[i], path_to_link_keeper)
            else:
                #print(links[i], link_type)
                raise Exception("Webpage is not txt, htm, or pdf")
        except:
            # write failed links
            failed_link_keeper = "Failed_Links.txt"
            failed_path = os.path.join(rule_path, failed_link_keeper)
            record_link(links[i], failed_path)
            

def update_binder(rule_path, this_wk_binder_path, this_wk_binder_name, latest_binder):
    #latest_binder = find_latest_binder(rule_path)
    temp_pdf_dir = os.path.join(this_wk_binder_path, "Temp_PDFs")
    temp_pdfs = os.listdir(temp_pdf_dir)
    temp_pdfs_filepaths = [os.path.join(temp_pdf_dir, pdf) for pdf in temp_pdfs]
    
    outline_dir = os.path.join(this_wk_binder_path, "Outlined_PDFs")
    if not os.path.exists(outline_dir):
        os.mkdir(outline_dir)
    
    for k in range(len(temp_pdfs_filepaths)):
        commenter_name = temp_pdfs_filepaths[k].split("/")[-1].split("_")[1]   
        commenter_filename = temp_pdfs_filepaths[k].split("/")[-1]
        pdf_newpath = os.path.join(outline_dir, commenter_filename)
        writer = PdfWriter()
        reader = PdfReader(temp_pdfs_filepaths[k])
        writer.add_page(reader.pages[0])
        writer.add_outline_item(commenter_name, 0, parent=None)

        for p in range(1, len(reader.pages)):
            writer.add_page(reader.pages[p])
        with open(pdf_newpath, 'wb') as fp:
            writer.write(fp)
        os.remove(temp_pdfs_filepaths[k])
    os.rmdir(temp_pdf_dir)
    
    outlined_pdfs = os.listdir(outline_dir)
    outlined_pdfs_fp = [os.path.join(outline_dir, pdf) for pdf in outlined_pdfs]
    outlined_pdfs_fp.extend(latest_binder)
    
    merger = PdfMerger(strict=False)
    for i in range(len(outlined_pdfs_fp)):

        merger.append(outlined_pdfs_fp[i], import_outline=True)
    merger_filename = f"{this_wk_binder_name}.pdf"
    merger_filepath = os.path.join(this_wk_binder_path, merger_filename)
    merger.write(merger_filepath)
    merger.close()
    
def find_latest_folder(rule_path):
    files_in_dir = os.listdir(rule_path)
    # list sub dirs if sub dir isn't Temp_PDFs
    sub_dirs = [os.path.join(rule_path, sd) for sd in files_in_dir if os.path.isdir(os.path.join(rule_path,sd))
                and "Temp_PDFs" not in sd]
    # get the subdir that was most recently modified
    try:
        latest_subdir = max(sub_dirs, key=os.path.getmtime)
    # this means there were no sub_dirs. This might be the first time running the program on a rule.
    except ValueError:
        latest_subdir = None
    return latest_subdir

# locates most recently modified folder and gets the PDF binder within it
def find_latest_binder(rule_path):
    latest_subdir = find_latest_folder(rule_path)
    # list stuff in rule_path
#     files_in_dir = os.listdir(rule_path)
#     # list sub dirs if sub dir isn't Temp_PDFs
#     sub_dirs = [os.path.join(rule_path, sd) for sd in files_in_dir if os.path.isdir(os.path.join(rule_path,sd))
#                 and "Temp_PDFs" not in sd]
#     # get the subdir that was most recently modified
#     latest_subdir = max(sub_dirs, key=os.path.getmtime)
    # a wild card to give to glob to find PDF binder
    if latest_subdir != None:
        wild_card = os.path.join(latest_subdir, "*Binder*.pdf")
        latest_binder = glob.glob(wild_card)
        return latest_binder
    else:
        return []

# returns csv from most recently modified folder
def find_latest_tracker(rule_path):
    latest_subdir = find_latest_folder(rule_path)
    wild_card = os.path.join(latest_subdir, "*.csv")
    latest_tracker = glob.glob(wild_card)
    return latest_tracker

def pdf_filenamer(name, link, date):
    try:
        # convert date to year month day
        ymd = datetime.strptime(date, "%b. %d, %Y").strftime("%Y-%m-%d")
        # make sure "/" isn't in the filename as this causes errors
        filename = f"{ymd}_{name}".replace("/","")
        # ensure filename is less than 250 characters
        filename = filename[:245] + ".pdf"
    except:
        filename = f"None_{name}".replace("/","")
        filename = filename[:245] + ".pdf"
    return filename
    
def single_page_scraper_v2(rule_page_link):
    #rule_page_link = params_dict['RULE_PAGE_LINK']
    comment_letters_directory = os.path.join(os.getcwd(), "Comment Letters")
    
    if not os.path.exists(comment_letters_directory):
        os.mkdir(comment_letters_directory)
    
    comment_page = requests.get(rule_page_link)
    comment_soup = BeautifulSoup(comment_page.content, "html.parser")
    rule_header1 = str(comment_soup.find('h1').string).strip('\n')
    rule_header2 = str(comment_soup.find('h2').string)
    
    rule_name = rule_nickname_lookup(rule_header1)
    if rule_page_link == 'https://www.sec.gov/comments/s7-20-22/s72022.htm':
        rule_name = 'Shareholder Proposals Rule 14a-8'
    file_no = rule_header2.split('File No.')[1].strip().strip(']')
    
    names, links, dates = rule_comment_info_retriever(rule_page_link)
    
    #hyperlinks = make_hyperlink(links, names)
    data = {'Date of Receipt': dates,
            'Letter Author': names,
            'URL': links}
    try:
        df = pd.DataFrame.from_dict(data)
    except:
        raise Exception("Cannot convert data dictionary to pandas df")
        #return data
    
    rule_name_file_no = f"{rule_name}_{file_no}"
    rule_path = os.path.join(comment_letters_directory, rule_name_file_no)
    
    if not os.path.exists(rule_path):
        os.mkdir(rule_path)
    
    # a txt file to keep track of links previously scraped in the past
    path_to_link_keeper = os.path.join(rule_path, "Download_Links.txt")
    
    # we only want to worry about comments we haven't downloaded
    undownloaded_links, undownloaded_names, undownloaded_dates = undownloaded_links_names(links,
                                                                names, dates, path_to_link_keeper)

    today = str(date.today().strftime("%m%d%Y"))
    delta_week = timedelta(days=7)
    last_week = str(date.today() - delta_week)
    last_week_date = datetime.strptime(last_week, '%Y-%m-%d').strftime("%m%d%Y")

    this_wk_binder_name = f"{rule_name_file_no}_Binder {today}"
    last_wk_binder_name = f"{rule_name_file_no}_Binder {last_week_date}"
    
    file_no_rule_name = f"{file_no}_{rule_name}"
    
    this_excel_filename = f"{file_no_rule_name}_Tracker {today}.csv"
    last_excel_filename = f"{file_no_rule_name}_Tracker {last_week_date}.csv"
    
#     this_excel_filepath = os.path.join(rule_path, this_excel_filename)
#    last_excel_filepath = os.path.join(rule_path, last_excel_filename)
    
    # if we downloaded everything already then just rename the tracker, binder, and folder
    if len(undownloaded_links) == 0:
        s = f"{rule_name} contains no new comments."
        print(s)        
        src = find_latest_folder(rule_path)
        dst = os.path.join(rule_path, this_wk_binder_name)
        
        latest_csv_filepath = glob.glob(os.path.join(dst, "*Tracker*.csv"))
        rename_csv_filepath = os.path.join(dst, this_excel_filename)
        os.rename(latest_csv_filepath[0], rename_csv_filepath)
        
        latest_binder = glob.glob(os.path.join(dst, "*Binder*.pdf"))
        rename_binder = os.path.join(dst, this_wk_binder_name)
        os.rename(latest_binder[0], rename_binder)
        
        os.rename(src, dst)
        
        return None
    
    this_wk_binder_path = os.path.join(rule_path, this_wk_binder_name)
    
    if not os.path.exists(this_wk_binder_path):
        # latest pdf binder is used to update binder
        # uses modified date to find latest subdir in rule path. Then get PDF binder that's there
        latest_binder = find_latest_binder(rule_path)
        os.mkdir(this_wk_binder_path)

    this_wk_tracker_path = os.path.join(this_wk_binder_path, this_excel_filename)
    df.to_csv(this_wk_tracker_path, index=False)
    
    download_webpage_v2(undownloaded_links, undownloaded_names, undownloaded_dates,
                                this_wk_binder_path, this_wk_binder_name, rule_path)
    
    #combine_link_txts(rule_path)
    temporary_pdf_directory = os.path.join(this_wk_binder_path, "Temp_PDFs")

    wild_card_dir = os.path.join(temporary_pdf_directory, "*.pdf")

    # get recently downloaded PDF files
    pdf_file_list = glob.glob(wild_card_dir)
    # append latest PDF binder
    pdf_file_list.extend(latest_binder)
    #this_wk_binder_path = os.path.join(directory, this_wk_binder_name)
    merged_pdf_filename = this_wk_binder_name+".pdf"
    merged_pdf_filepath = os.path.join(this_wk_binder_path, merged_pdf_filename)
    # sort them
    pdf_file_list = sorted(pdf_file_list)
    
    # There are 4000+ comments in climate. Attempting to merge 4000+ PDF files results in OSErrors
    # So I open 1000 PDF files at a time, merge them, and save into `num_files/1000` merged PDF files
    # then I take those merged PDF files and create a master merged PDF file containing everything

    # if this is the first time a rule is scraped then start from scratch
    if len(latest_binder) == 0:
        climate_outliner_v1(rule_path)
    else:
        # otherwise, update the binder
        update_binder(rule_path, this_wk_binder_path, this_wk_binder_name, latest_binder)
        
    s = f"{rule_name} contains {len(undownloaded_links)} new comments."
    print(s)
    return df