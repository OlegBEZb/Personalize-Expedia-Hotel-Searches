from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from pandas.tseries.holiday import (HolidayCalendarFactory,
                                    Holiday, AbstractHolidayCalendar,
                                    MO, TU, SU, TH, nearest_workday)


HOL_FLAG_PREFIX = "hol_flag_"

# region Holidays

class MyHoliday(Holiday):
    def __init__(self, name, year=None, month=None, day=None, offset=None,
                 observance=None, special_rules=None, start_date=None, end_date=None, subhols=None):
        """Uses the Holiday parent logic to handle offsets and then overrides the observance function to support extra logic on the offset.
        Also adds the subhols,  a list of (holiday_name, offset).
        """
        super(MyHoliday, self).__init__(name, year=year, month=month, day=day, offset=offset,
                                        start_date=start_date, end_date=end_date, )
        self._observance = special_rules
        self.subhols = subhols

    def __repr__(self):
        res = Holiday.__repr__(self)
        res = 'My' + res
        res += ', special_rules = {}, subhols= {}'.format(self._observance, self.subhols)
        return res

    def _apply_rule(self, dates):
        """
        Apply the given offset/observance to an
        iterable of dates.

        Parameters
        ----------
        dates : array-like
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates of the main holiday with rules applied.
        Sub hols will be added to this list later
        """
        dates = Holiday._apply_rule(self, dates)
        if self._observance is not None:
            dates = dates.map(lambda d: self._observance(d)).dropna()
        return dates

    def dates(self, start_date, end_date, return_name=False):
        """The core function in build holidays."""
        res = Holiday.dates(self, start_date, end_date, return_name)
        if not self.subhols:
            return res
        if return_name:
            dts = res.index.copy()
        else:
            dts = res.copy()

        # build the list of other holidays
        extras = [(dt + Day(offset), name) for name, offset in self.subhols for dt in dts]
        if return_name:
            return res.append(pd.Series(data=[e[1] for e in extras], index=[e[0] for e in extras])).sort_index()
        else:
            return res.append(pd.DatetimeIndex(data=[e[0] for e in extras])).sort()


def generate_holidays(d_start, d_end, a_dt_column="dt", a_hol_column="hol_dominant"):
    start_date, end_date = d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d")

    def skip_odd_years(dt):
        return pd.NaT if dt.year % 2 == 1 else dt

    hn = []
    new_year_hol_name = "New Year's Day"; hn.append(new_year_hol_name)
    jan_sales_slump_hol_name = "January sales slump (Jan 4-20)"; hn.append(jan_sales_slump_hol_name)
    mlk_hol_name = "Dr. Martin Luther King Jr. Day"; hn.append(mlk_hol_name)
    val_hol_name = "Valentine's Day"; hn.append(val_hol_name)
    pres_day_hol_name = "Presidents' Day"; hn.append(pres_day_hol_name)
    ash_hol_name = "Ash Wednesday"; hn.append(ash_hol_name)
    friday_during_lent_hol_name = "Fridays during lent"; hn.append(friday_during_lent_hol_name)
    good_friday_hol_name = "Good Friday"; hn.append(good_friday_hol_name)
    easter_sunday_hol_name = "Easter Sunday"; hn.append(easter_sunday_hol_name)
    ptk_hol_name = "St. Patrick's Day"; hn.append(ptk_hol_name)
    mothers_day_hol_name = "Mother's Day"; hn.append(mothers_day_hol_name)
    memorial_day_hol_name = "Memorial Day"; hn.append(memorial_day_hol_name)
    fathers_day_hol_name = "Father's Day"; hn.append(fathers_day_hol_name)
    ind_hol_name = "Independence Day"; hn.append(ind_hol_name)
    july4_federal_hol_name = "July 4th Federal"; hn.append(july4_federal_hol_name)
    labor_day_hol_name = "Labor Day"; hn.append(labor_day_hol_name)
    columbus_day_hol_name = "Columbus Day"; hn.append(columbus_day_hol_name)
    halloween_hol_name = "Halloween"; hn.append(halloween_hol_name)
    election_day_hol_name = "Election Day"; hn.append(election_day_hol_name)
    wednesday_before_thanksgiving_hol_name = "Wednesday before Thanksgiving"; hn.append(wednesday_before_thanksgiving_hol_name)
    thanksgiving_day_hol_name = "Thanksgiving Day"; hn.append(thanksgiving_day_hol_name)
    black_friday_hol_name = "Black Friday"; hn.append(black_friday_hol_name)
    day_after_black_friday_hol_name = "Day after Black Friday"; hn.append(day_after_black_friday_hol_name)
    veterans_day_hol_name = "Veterans Day"; hn.append(veterans_day_hol_name)
    christmas_eve_hol_name = "Christmas Eve"; hn.append(christmas_eve_hol_name)
    christmas_day_hol_name = "Christmas Day"; hn.append(christmas_day_hol_name)
    new_years_eve_hol_name = "New Year's Eve"; hn.append(new_years_eve_hol_name)

    hols = [
               MyHoliday(new_year_hol_name, month=1, day=1)
           ] + [
               MyHoliday(f'[d={d}]{jan_sales_slump_hol_name}', month=1, day=1, offset=pd.DateOffset(day=d))
               for d in range(4, 20 + 1)
           ] + [
               MyHoliday(mlk_hol_name, month=1, day=1, offset=pd.DateOffset(weekday=MO(+3))),
               MyHoliday(val_hol_name, month=2, day=14),
               Holiday(pres_day_hol_name, month=2, day=1, offset=pd.DateOffset(weekday=MO(+3))),
               MyHoliday(easter_sunday_hol_name, month=1, day=1, offset=Easter(),
                         subhols=[(good_friday_hol_name, -2)] + [(friday_during_lent_hol_name, -2 - 7 * i) for i in range(1, 6 + 1)] + [(ash_hol_name, -46)]),
               Holiday(ptk_hol_name, month=3, day=17),
               Holiday(mothers_day_hol_name, month=5, day=1, offset=pd.DateOffset(weekday=SU(+2))),
               Holiday(memorial_day_hol_name, month=5, day=24, offset=pd.DateOffset(weekday=MO(+1))),
               Holiday(fathers_day_hol_name, month=6, day=1, offset=pd.DateOffset(weekday=SU(+3))),
               MyHoliday(ind_hol_name, month=7, day=4),
               Holiday(july4_federal_hol_name, month=7, day=4, observance=nearest_workday),
               Holiday(labor_day_hol_name, month=9, day=1, offset=pd.DateOffset(weekday=MO(+1))),
               Holiday(columbus_day_hol_name, month=10, day=1, offset=pd.DateOffset(weekday=MO(+2))),
               MyHoliday(halloween_hol_name, month=10, day=31),
               MyHoliday(election_day_hol_name, month=11, day=1, offset=pd.DateOffset(weekday=TU(+1)), special_rules=skip_odd_years),
               MyHoliday(thanksgiving_day_hol_name, month=11, day=1, offset=pd.DateOffset(weekday=TH(+4)),
                         subhols=[(wednesday_before_thanksgiving_hol_name, -1),
                                  (black_friday_hol_name, 1),
                                  (day_after_black_friday_hol_name, 2)]
                         ),
               Holiday(veterans_day_hol_name, month=11, day=11),
               MyHoliday(christmas_eve_hol_name, month=12, day=24),
               MyHoliday(christmas_day_hol_name, month=12, day=25),
               MyHoliday(new_years_eve_hol_name, month=12, day=31),
           ]

    hols_factory = HolidayCalendarFactory('My Holidays', AbstractHolidayCalendar(), hols)
    pdf_holidays = pd.DataFrame(hols_factory().holidays(start=start_date, end=end_date, return_name=True), columns=[a_hol_column])
    pdf_holidays = pdf_holidays.reset_index().rename(columns={"index": a_dt_column})

    def fix_hol_name(a_name):
        if a_name.startswith("["):
            i = a_name.index("]")
            return a_name[i + 1:]
        return a_name

    pdf_holidays[a_hol_column] = pdf_holidays[a_hol_column].map(fix_hol_name)

    # before keeping the dominant holiday, let us first build a pivot with indicators
    pdf_holidays["c"] = 1
    pdf_holidays_pivot = pdf_holidays.pivot(a_dt_column, a_hol_column, "c")
    def is_hol_col_name(a_column):
        return a_column.lower().replace(" ", "_").replace("-", "_").replace("'", "").replace("(", "").replace(")", "").replace(".", "")

    hi_names = [HOL_FLAG_PREFIX + is_hol_col_name(h) for h in hn]
    # first we neeed to change the column names in pivot
    pdf_holidays_pivot.columns = [HOL_FLAG_PREFIX + is_hol_col_name(c) for c in pdf_holidays_pivot.columns]

    # convention is next:  all dataframe (no matter of date ranges) must have ALL holiday fields
    for h in hi_names:
        if h not in pdf_holidays_pivot.columns:
            pdf_holidays_pivot[h] = 0

    # bad assert: if we select NOT whole year, then we should not expect all the list here
    # assert len(pdf_holidays_pivot.columns) == len(hi_names), "Some field is missing in the list"
    pivot_cols_set = set(pdf_holidays_pivot.columns)
    assert len(pivot_cols_set - set(hi_names)) == 0, f"Wrong names of pivot columns: {pivot_cols_set - set(hi_names)}"
    #hi_selected_names = [h for h in hi_names if h in pivot_cols_set]  # order is different: we must take it in order of hi_nam,es
    # select them in correct order
    pdf_holidays_pivot = pdf_holidays_pivot[hi_names]

    pdf_holidays_pivot.fillna(0, inplace=True)
    for c in pdf_holidays_pivot.columns:
        if c.startswith(HOL_FLAG_PREFIX):
            pdf_holidays_pivot[c] = pdf_holidays_pivot[c].astype(np.int8)
    pdf_holidays_pivot.reset_index(inplace=True)

    # reason for doing this code is this:
    # 1) Dr. Martin Luther King Jr. can intersect with January sales slump(Jan 4-20)
    # 2) St. Patrick's Day may be on Driday when the Lent Friday is
    # 3) Independence day can be a working dat
    # 4) Valentine's day can match with Ash Wednesday 
    for dominant_holiday in [mlk_hol_name, ptk_hol_name, ind_hol_name, ash_hol_name]:
        date_of_dominant = set(pdf_holidays[pdf_holidays[a_hol_column] == dominant_holiday][a_dt_column].values)
        pdf_holidays = pdf_holidays[
            (~pdf_holidays[a_dt_column].isin(date_of_dominant))
            |
            ((pdf_holidays[a_dt_column].isin(date_of_dominant)) & (pdf_holidays[a_hol_column] == dominant_holiday))
            ]

    pdf_test = pdf_holidays[[a_dt_column, "c"]].groupby(a_dt_column).count()
    assert pdf_test[pdf_test["c"] > 1].shape[0] == 0, "There must be no duplicates after we've removed matching holidays!"
    pdf_holidays.drop(columns=["c"], inplace=True)

    return pd.merge(pdf_holidays, pdf_holidays_pivot, how="inner", on=a_dt_column).sort_values(a_dt_column)


def generate_holidays_for_year_range(a_start_year, a_end_year, a_dt_column="dt", a_hol_column="hol_dominant"):
    d_start = datetime(a_start_year, 1, 1)
    d_start = d_start + timedelta(days=-d_start.weekday())
    d_end = datetime(a_end_year, 12, 31)
    d_end = d_end + timedelta(days=-d_end.weekday() + 6)
    return generate_holidays(d_start, d_end, a_dt_column=a_dt_column, a_hol_column=a_hol_column)

# endregion
