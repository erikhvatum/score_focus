from pathlib import Path
import sqlite3

with sqlite3.connect('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/analysis/db.sqlite3') as db:
    db.row_factory = sqlite3.Row
    hatched_well_idxs = [row['well_idx'] for row in db.execute('select well_idx from wells where did_hatch=1')]
    time_points = [row['name'] for row in db.execute('select name from time_points')]
    for time_point in time_points:
        for well_idx in hatched_well_idxs:
            manual_focus_scores_row = list(db.execute('select * from manual_focus_scores where well_idx=? and time_point=?', (well_idx, time_point)))
            if manual_focus_scores_row:
                manual_focus_scores_row = manual_focus_scores_row[0]
                if manual_focus_scores_row['has_bf']:
                    db.execute(
                        'insert into images (well_idx, time_point, acquisition_name, is_focused) values (?, ?, ?, ?)',
                        (well_idx, time_point, 'bf', manual_focus_scores_row['bf_is_focused']))
                for n in range(manual_focus_scores_row['focus_stack_len']):
                    db.execute(
                        'insert into images (well_idx, time_point, acquisition_name, is_focused) values (?, ?, ?, ?)',
                        (well_idx, time_point, 'focus-{:02}'.format(n), manual_focus_scores_row['best_focus_stack_idx']==n))
    db.commit()
