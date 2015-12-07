# The MIT License (MIT)
#
# Copyright (c) 2015 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

from pathlib import Path
from PyQt5 import Qt
import sqlite3

class FocusScores(Qt.QObject):
    def __init__(self, db, wellIdx, timePoint):
        super().__init__()
        self.db = db
        self._wellIdx = wellIdx
        self._timePoint = timePoint

    @Qt.pyqtProperty(int)
    def wellIdx(self):
        return self._wellIdx

    @wellIdx.setter
    def setWellIdx(self, wellIdx):
        self._wellIdx = wellIdx

    @Qt.pyqtProperty(Qt.QDateTime)
    def timepoint(self):
        return self._timePoint

    @timepoint.setter
    def setTimepoint(self, timePoint):
        self._timePoint = timePoint

class ExperimentManualFocusScorer(Qt.QQuickItem):
    isValidChanged = Qt.pyqtSignal(bool)
    experimentDPathChanged = Qt.pyqtSignal(str)
    focusedWellIdxsChanged = Qt.pyqtSignal(Qt.QQmlListProperty)
    timePointsChanged = Qt.pyqtSignal(Qt.QQmlListProperty)
    focusScoresChanged = Qt.pyqtSignal(FocusScores)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rw = None
        self._db = None
        self._experimentDPath = ''
        self._focusedWellIdxs = []
        self._timePoints = []

    @Qt.pyqtProperty(bool, notify=isValidChanged)
    def isValid(self):
        return self.db is not None

    @Qt.pyqtProperty(str, notify=experimentDPathChanged)
    def experimentDPath(self):
        return str(self._experimentDPath)

    @experimentDPath.setter
    def setExperimentDPath(self, experimentDPath):
        experimentDPath = Path(experimentDPath)
        db = sqlite3.connect(str(experimentDPath / 'analysis' / 'db.sqlite3'))
        self.db.row_factory = sqlite3.Row
        self._focusedWellIdxs = [row['well_idx'] for row in self.db.execute('select well, did_hatch from wells') if row['did_hatch']]
        self._timePoints = [row['name'] for row in self.db.execute('select name from time_points')]

    @Qt.pyqtProperty(Qt.QQmlListProperty, notify=focusedWellIdxsChanged)
    def focusedWellIdxs(self):
        return self._focusedWellIdxs

    @Qt.pyqtProperty(Qt.QQmlListProperty, notify=timePointsChanged)
    def timePoints(self):
        return self._timePoints

def _register_qml_types():
    Qt.qmlRegisterType(FocusScores, 'Analysis', 1, 0, 'FocusScores')
    Qt.qmlRegisterType(ExperimentManualFocusScorer, 'Analysis', 1, 0, 'ExperimentManualFocusScorer')

def make_as_rw_dock_widget(rw):
    rw.experiment_manual_focus_scorer_dock_widget = Qt.QDockWidget('Experiment Manual Focus Scorer', rw)
    rw.experiment_manual_focus_scorer_qml_container = Qt.QQuickWidget()
    rw.experiment_manual_focus_scorer_qml_container.resize(300,300)
    rw.experiment_manual_focus_scorer_qml_container.setResizeMode(Qt.QQuickWidget.SizeRootObjectToView)
    rw.experiment_manual_focus_scorer_qml_container.setSource(Qt.QUrl(str(Path(__file__).parent / 'manually_score_focus.qml')))
    rw.experiment_manual_focus_scorer_qml_container.rootContext().setContextProperty("backgroundColor", Qt.QColor(Qt.Qt.red))
    rw.experiment_manual_focus_scorer_dock_widget.setWidget(rw.experiment_manual_focus_scorer_qml_container)
    rw.experiment_manual_focus_scorer_dock_widget.setAllowedAreas(Qt.Qt.RightDockWidgetArea | Qt.Qt.LeftDockWidgetArea)
    rw.experiment_manual_focus_scorer_dock_widget.setFeatures(
        Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
    rw.addDockWidget(Qt.Qt.RightDockWidgetArea, rw.experiment_manual_focus_scorer_dock_widget)
    rw.dock_widget_visibility_toolbar.addAction(rw.experiment_manual_focus_scorer_dock_widget.toggleViewAction())
    rw.experiment_manual_focus_scorer = rw.experiment_manual_focus_scorer_qml_container.rootObject()
    Qt.QQmlProperty(rw.experiment_manual_focus_scorer, 'backgroundColor').write(
        Qt.QVariant(Qt.QColor(239, 239, 239)))#Qt.QApplication.instance().palette.color(Qt.QPalette.Window)))
    return rw.experiment_manual_focus_scorer

if __name__ == "__main__":
    from ris_widget.ris_widget import RisWidget
    import argparse
    parser = argparse.ArgumentParser(description='Manually score BF image and focus stacks.')
    parser.add_argument(
        '--experiment_dpath',
        type=Path,
        help='Directory containing experiment_metadata.json.  Default: "/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3"',
        default=Path('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3'),
        required=False)
    args = parser.parse_args()
    app_obj_args = []
    app = Qt.QApplication(app_obj_args)
    _register_qml_types()
    rw = RisWidget(msaa_sample_count=8)
    manually_score_focus = make_as_rw_dock_widget(rw.qt_object)
    # manually_score_focus.setExperimentDPath(Path(__file__))
    rw.show()
    app.exec()
else:
    _register_qml_types()