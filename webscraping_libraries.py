# webscraping libraries
import requests
from bs4 import BeautifulSoup

# PDF manipulation libraries
from pyhtml2pdf import converter
from txt2pdf.core import txt2pdf
from pdfy import Pdfy
from pikepdf import Pdf
from pypdf import PdfMerger, PdfReader, PdfWriter

# file handling
import glob
import os
import shutil

# data manipulation
import pandas as pd
import numpy as np

# concurrency 
import concurrent.futures
import threading

# misc
import argparse
from datetime import date, timedelta, datetime
import tqdm
import psutil