from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
import re

def expand_norwegian_months(text):
    month_mapping = {
        "jan": "januar", "feb": "februar", "mar": "mars", "apr": "april",
        "mai": "mai", "jun": "juni", "jul": "juli", "aug": "august",
        "sep": "september", "okt": "oktober", "nov": "november", "des": "desember"
    }
    
    def replace_match(match):
        day1, short_month, day2 = match.groups()
        long_month = month_mapping.get(short_month.lower(), short_month)
        return f"{day1} {long_month} {day2}"
    
    pattern = r"(\d{1,2})\s*(jan|feb|mar|apr|mai|jun|jul|aug|sep|okt|nov|des)\s*(\d{1,2})"
    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

def dct_year(dct, month, day):
    """ If event date is in the future of DCT, assign it to the previous year """
    current_year = dct.year
    event_date = datetime(current_year, int(month), int(day))

    if event_date >= dct:
        current_year -= 1
        
    return current_year

def expand_negative_year(match):
    """ Converts -YY to 1.1.YYYY """
    year = match.group(1)
    full_year = f"20{year}" if int(year) < 30 else f"19{year}"
    return f"1.1.{full_year}"

def convert_negative_years(text):
    """ Finds and replaces -YY with 1.1.YYYY """
    pattern = r"-\b(\d{2})\b"
    return re.sub(pattern, expand_negative_year, text)

def expand_partial_date(match, dct):
    """ Converts DD.MM or DD.M or D.M to DD.MM.YYYY based on DCT logic """
    day, month = match.groups()
    
    # Ensure the month has two digits
    month = month.zfill(2)
    
    current_year = dct_year(dct, month, day)

    return f"{day}.{month}.{current_year}"

def convert_partial_dates(text, dct):
    """ Finds and replaces DD.MM or DD.M or D.M with DD.MM.YYYY based on DCT """
    # Pattern to match DD.MM or DD.M formats
    pattern = r"\b(\d{1,2})\.(\d{1,2})\b(?!\.\d{2})"  # Matches DD.MM or DD.M format
    pattern = r"(?<!kl\.)\b(\d{1,2})\.(\d{1,2})\b(?!\.\d{2})"
    text = text.replace('kl. ', 'kl.')
    return re.sub(pattern, lambda match: expand_partial_date(match, dct), text)

def convert_full_year(date_str):
    """ Convert full year YYYY to 1.1.YYYY"""
    # Regex to match dates in the format DD.MM.YYYY or D.M.YYYY
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
        return date_str  # No conversion, return as is
    # If it's just YYYY, convert to 1.1.YYYY
    elif re.match(r'^\d{4}$', date_str):
        if int(date_str) < 1850:
            return date_str
        return f"1.1.{date_str}"
    # In case of any other invalid format, return the input as is
    return date_str

def convert_date_format_2(text):
    # Updated regex to match a valid 6-digit date (DDMMYY) and ensure no other numbers follow directly
    match = re.search(r"\b(\d{2}\d{2}\d{2})\b", text)
    
    if match:
        date_str = match.group(1)  # Extract the found date
        try:
            date_obj = datetime.strptime(date_str, "%d%m%y")  # Convert to date object
            formatted_date = date_obj.strftime("%d.%m.%y")  # Format as DD.MM.YY
            
            # Replace the original date in text with formatted date
            text = text.replace(date_str, formatted_date)
        except ValueError as e:
            print(f"Error parsing date: {e}")
    
    return text

def convert_date_format(text):
    """ Convert DD/MM/YYYY to DD.MM.YYYY"""
    
    # Regular expression to match dates in different formats
    date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{2})(?!\d)'  # Matches D.M.YY, DD.M.YY, D.MM.YY, DD.MM.YY, but not DD.MM.YYYY
    def format_date(match):
        day = int(match.group(1))  # Day part
        month = int(match.group(2))  # Month part
        year = match.group(3)  # Year part
        year = f"20{year}" if int(year) < 30 else f"19{year}"
        
        # Return in the desired format
        return f'{day:02d}.{month:02d}.{year}'
    
    # Replace all dates in the text using the format_date function
    return re.sub(date_pattern, format_date, text)

def convert_date_format_3(text):
    def convert_date(match):
        month, day, year = match.groups()
        # Normalize year to 4 digits
        if len(year) == 2:
            year = f"20{year}" if int(year) < 30 else f"19{year}"
            return f"{int(day):02d}.{int(month):02d}.{year}"

    # Updated pattern: month/day-year (e.g. 11/9-1985)
    pattern = r'(\d{1,2})/(\d{1,2})-(\d{2,4})'

    new_text = re.sub(pattern, convert_date, text)

    return new_text

def convert_text(text):
    text = text.replace("i dag", "idag")
    text = text.replace("mnd", "måned")
    text = text.replace("månedr", "måneder")
    text = text.replace("på", "")
    text = text.replace("imorgen", "i morgen")
    return text

def convert_slash_date(text):
    # Regular expression to match dates in the various given formats
    date_pattern = r'(\d{1,2})/(\d{1,2})/(\d{2,4})'

    def format_date(match):
        day, month, year = match.groups()
        year = f"20{year}" if int(year) < 30 else f"19{year}"
        # Ensure day and month are two digits
        return f'{int(day):02d}.{int(month):02d}.{int(year):04d}'

    # Find all dates in the text using the pattern and replace them
    return re.sub(date_pattern, format_date, text)

def convert_duration(context, value, dct):
    time_units = {"M": relativedelta(months=1), "W": timedelta(weeks=1), "D": timedelta(days=1), "Y": relativedelta(years=1)}
    unit = next((unit for unit in time_units if unit in value), None)
    
    if unit:
        print("UNIT", unit)
        try:
            amount = int(value.replace("P", "").replace(unit, ""))
        except:
            amount = 0
        adjustment = time_units[unit]
        resolved_date = dct
        if any(term in context for term in ["siden", "tilbake", "i forveien", "før"]):  
            resolved_date = dct - adjustment * amount
        elif any(term in context for term in ["etter", "om", "til", "senere", "postoperativ"]):
            resolved_date = dct + adjustment * amount
        resolved_date = resolved_date.strftime("%Y-%m-%d")
    else:
        resolved_date = value
    return resolved_date

