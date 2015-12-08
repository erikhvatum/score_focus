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

import freeimage
from pathlib import Path
from PyQt5 import Qt
import sqlite3

class SimpleListModel(Qt.QAbstractListModel):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def rowCount(self, parent=None):
        return len(self.data)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid():
            return Qt.QVariant(str(self.data[midx.row()]))
        if role == Qt.Qt.DisplayRole and midx.isValid():
            return Qt.QVariant(str(self.data[midx.row()]))
        return super().data(midx, role)

class ManualFocusScore(Qt.QObject):
    wellIdxChanged = Qt.pyqtSignal(int)
    timepointChanged = Qt.pyqtSignal(str)
    hasBfChanged = Qt.pyqtSignal(bool)
    bfIsFocusedChanged = Qt.pyqtSignal(bool)
    focusStackLenChanged = Qt.pyqtSignal(int)
    bestFocusStackIdxChanged = Qt.pyqtSignal(int)

    def __init__(self, experimentalManualFocusScorer):
        super().__init__()
        self.experimentalManualFocusScorer = experimentalManualFocusScorer
        self._wellIdx = -1
        self._timePoint = ""
        self._hasBf = False
        self._bfIsFocused = False
        self._focusStackLen = 0
        self._bestFocusStackIdx = -1

    @Qt.pyqtProperty(int, notify=wellIdxChanged)
    def wellIdx(self):
        return self._wellIdx

    @wellIdx.setter
    def wellIdx(self, wellIdx):
        if wellIdx != self._wellIdx:
            if wellIdx not in self.experimentalManualFocusScorer._hatchedWellIdxs.data:
                wellIdx = -1
            self._wellIdx = wellIdx
            self.showCurrentImage()
            self.refresh()
            self.wellIdxChanged.emit(self._wellIdx)

    def showCurrentImage(self):
        if self._wellIdx != -1 and self._timePoint:
            wellDPath = self.experimentalManualFocusScorer._experimentDPath / '{:02}'.format(self._wellIdx)
            if self._bestFocusStackIdx == -1:
                imageFPath = wellDPath / '{} bf.png'.format(self._timePoint)
            else:
                imageFPath = wellDPath / '{} focus-{:02}.png'.format(self._timePoint, self._bestFocusStackIdx)
            rw.image = freeimage.read(str(imageFPath))

    @Qt.pyqtProperty(str)
    def timepoint(self, notify=timepointChanged):
        return self._timePoint

    @timepoint.setter
    def timepoint(self, timePoint):
        if self._timePoint != timePoint:
            if timePoint not in self.experimentalManualFocusScorer._timePoints.data:
                timePoint = ""
            self._timePoint = timePoint
            self.showCurrentImage()
            self.refresh()
            self.timepointChanged.emit(self._timePoint)

    @Qt.pyqtSlot()
    def refresh(self):
        if self._wellIdx != -1 and self._timePoint:
            q = list(
                self.experimentalManualFocusScorer.db.execute(
                    'select has_bf, bf_is_focused, focus_stack_len, best_focus_stack_idx from manual_focus_scores where well_idx=? and time_point=?',
                    (self._wellIdx, self._timePoint)
                )
            )
            if q:
                self._setHasBf(q[0]['has_bf'])
                self.bfIsFocused = q[0]['bf_is_focused']
                self._setFocusStackLen(q[0]['focus_stack_len'])
                self.bestFocusStackIdx = q[0]['best_focus_stack_idx']
            else:
                wellDPath = self.experimentalManualFocusScorer._experimentDPath / '{:02}'.format(self._wellIdx)
                self._setHasBf((wellDPath / '{} bf.png'.format(self._timePoint)).exists())
                self.bfIsFocused = False
                stackFPaths = sorted(list(wellDPath.glob('{} focus-*.png'.format(self._timePoint))))
                if stackFPaths:
                    self._setFocusStackLen(int(str(stackFPaths[-1])[-6:-4]))
                else:
                    self._setFocusStackLen(0)
                self.bestFocusStackIdx = -1

    @Qt.pyqtSlot()
    def commit(self):
        if self._wellIdx != -1 and self._timePoint:
            if list(self.experimentalManualFocusScorer.db.execute('select * from manual_focus_scores where well_idx=? and time_point=?', (self._wellIdx, self._timePoint))):
                list(self.experimentalManualFocusScorer.db.execute(
                    'update manual_focus_scores set has_bf=?, bf_is_focused=?, focus_stack_len=?, best_focus_stack_idx=? where well_idx=? and time_point=?',
                    (self._hasBf, self._bfIsFocused, self._focusStackLen, self._bestFocusStackIdx, self._wellIdx, self._timePoint)))
            else:
                list(self.experimentalManualFocusScorer.db.execute(
                    'insert into manual_focus_scores (has_bf, bf_is_focused, focus_stack_len, best_focus_stack_idx, well_idx, time_point) values (?, ?, ?, ?, ?, ?)',
                    (self._hasBf, self._bfIsFocused, self._focusStackLen, self._bestFocusStackIdx, self._wellIdx, self._timePoint)))
            self.experimentalManualFocusScorer.db.commit()

    @Qt.pyqtSlot()
    def commitAndAdvance(self):
        if self._wellIdx != -1 and self._timePoint:
            try:
                wellIdxIdx = self.experimentalManualFocusScorer._hatchedWellIdxs.data.index(self._wellIdx)
            except ValueError:
                pass
            else:
                self.commit()
                if wellIdxIdx == len(self.experimentalManualFocusScorer._hatchedWellIdxs.data) - 1:
                    try:
                        timepointIdx = self.experimentalManualFocusScorer._timePoints.data.index(self._timePoint)
                    except ValueError:
                        pass
                    else:
                        if timepointIdx < len(self.experimentalManualFocusScorer._timePoints.data) - 1:
                            self.timepoint = self.experimentalManualFocusScorer._timePoints.data[timepointIdx + 1]
                            self.wellIdx = self.experimentalManualFocusScorer._hatchedWellIdxs.data[0]
                else:
                    self.wellIdx = self.experimentalManualFocusScorer._hatchedWellIdxs.data[wellIdxIdx + 1]

    @Qt.pyqtSlot()
    def advanceToNextUnscored(self):
        try:
            wellIdxIdx = self.experimentalManualFocusScorer._hatchedWellIdxs.data.index(self._wellIdx)
            timepointIdx = self.experimentalManualFocusScorer._timePoints.data.index(self._timePoint)
        except ValueError:
            return
        while True:
            if wellIdxIdx == len(self.experimentalManualFocusScorer._hatchedWellIdxs.data) - 1:
                if timepointIdx < len(self.experimentalManualFocusScorer._timePoints.data) - 1:
                    timepointIdx += 1
                    wellIdxIdx = 0
                else:
                    return
            else:
                wellIdxIdx += 1
            wellIdx = self.experimentalManualFocusScorer._hatchedWellIdxs.data[wellIdxIdx]
            timepoint = self.experimentalManualFocusScorer._timePoints.data[timepointIdx]
            if not list(self.experimentalManualFocusScorer.db.execute('select * from manual_focus_scores where well_idx=? and time_point=?', (wellIdx, timepoint))):
                self.wellIdx = wellIdx
                self.timepoint = timepoint
                return


    @Qt.pyqtProperty(bool, notify=hasBfChanged)
    def hasBf(self):
        return self._hasBf

    def _setHasBf(self, v):
        if self._hasBf != v:
            self._hasBf = v
            self.hasBfChanged.emit(self._hasBf)

    @Qt.pyqtProperty(bool, notify=bfIsFocusedChanged)
    def bfIsFocused(self):
        return self._bfIsFocused

    @bfIsFocused.setter
    def bfIsFocused(self, v):
        if self._wellIdx != -1 and self._timePoint and v != self._bfIsFocused:
            self._bfIsFocused = v
            self.bfIsFocusedChanged.emit(self._bfIsFocused)

    @Qt.pyqtProperty(int, notify=focusStackLenChanged)
    def focusStackLen(self):
        return self._focusStackLen

    def _setFocusStackLen(self, v):
        if self._focusStackLen != v:
            self._focusStackLen = v
            self.focusStackLenChanged.emit(self._focusStackLen)

    @Qt.pyqtProperty(int, notify=bestFocusStackIdxChanged)
    def bestFocusStackIdx(self):
        return self._bestFocusStackIdx

    @bestFocusStackIdx.setter
    def bestFocusStackIdx(self, v):
        if self._wellIdx != -1 and self._timePoint and v != self._bestFocusStackIdx:
            self._bestFocusStackIdx = v
            self.showCurrentImage()
            self.bestFocusStackIdxChanged.emit(self._bestFocusStackIdx)

class ExperimentManualFocusScorer(Qt.QQuickItem):
    isValidChanged = Qt.pyqtSignal(bool)
    experimentDPathChanged = Qt.pyqtSignal(str)
    hatchedWellIdxsChanged = Qt.pyqtSignal()
    timePointsChanged = Qt.pyqtSignal()
    manualFocusScoreChanged = Qt.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rw = None
        self.db = None
        self._experimentDPath = ''
        self._hatchedWellIdxs = SimpleListModel([])
        self._timePoints = SimpleListModel([])
        self._manualFocusScore = ManualFocusScore(self)

    def _init_actions(self):
        self.toPrevTimepointAction = Qt.QAction(self.rw)
        self.toPrevTimepointAction.setShortcut(Qt.Qt.Key_Minus)
        self.toPrevTimepointAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toPrevTimepointAction.triggered.connect(self.toPrevTimepoint)
        self.rw.addAction(self.toPrevTimepointAction)

        self.toNextTimepointAction = Qt.QAction(self.rw)
        self.toNextTimepointAction.setShortcut(Qt.Qt.Key_Equal)
        self.toNextTimepointAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toNextTimepointAction.triggered.connect(self.toNextTimepoint)
        self.rw.addAction(self.toNextTimepointAction)

        self.toPrevWellAction = Qt.QAction(self.rw)
        self.toPrevWellAction.setShortcut(Qt.Qt.Key_BracketLeft)
        self.toPrevWellAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toPrevWellAction.triggered.connect(self.toPrevWell)
        self.rw.addAction(self.toPrevWellAction)

        self.toNextWellAction = Qt.QAction(self.rw)
        self.toNextWellAction.setShortcut(Qt.Qt.Key_BracketRight)
        self.toNextWellAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toNextWellAction.triggered.connect(self.toNextWell)
        self.rw.addAction(self.toNextWellAction)

        self.uncheckBfFocusedAction = Qt.QAction(self.rw)
        self.uncheckBfFocusedAction.setShortcut(Qt.Qt.Key_Semicolon)
        self.uncheckBfFocusedAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.uncheckBfFocusedAction.triggered.connect(self.uncheckBfFocused)
        self.rw.addAction(self.uncheckBfFocusedAction)

        self.checkBfFocusedAction = Qt.QAction(self.rw)
        self.checkBfFocusedAction.setShortcut(Qt.Qt.Key_Apostrophe)
        self.checkBfFocusedAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.checkBfFocusedAction.triggered.connect(self.checkBfFocused)
        self.rw.addAction(self.checkBfFocusedAction)

        self.toPrevFocusStackIdxAction = Qt.QAction(self.rw)
        self.toPrevFocusStackIdxAction.setShortcut(Qt.Qt.Key_Period)
        self.toPrevFocusStackIdxAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toPrevFocusStackIdxAction.triggered.connect(self.toPrevFocusStackIdx)
        self.rw.addAction(self.toPrevFocusStackIdxAction)

        self.toNextFocusStackIdxAction = Qt.QAction(self.rw)
        self.toNextFocusStackIdxAction.setShortcut(Qt.Qt.Key_Slash)
        self.toNextFocusStackIdxAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.toNextFocusStackIdxAction.triggered.connect(self.toNextFocusStackIdx)
        self.rw.addAction(self.toNextFocusStackIdxAction)

        self.commitAndAdvanceAction = Qt.QAction(self.rw)
        self.commitAndAdvanceAction.setShortcut(Qt.Qt.Key_Backslash)
        self.commitAndAdvanceAction.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.commitAndAdvanceAction.triggered.connect(self.manualFocusScore.commitAndAdvance)
        self.rw.addAction(self.commitAndAdvanceAction)

    @Qt.pyqtProperty(bool, notify=isValidChanged)
    def isValid(self):
        return bool(self._hatchedWellIdxs.data) and bool(self._timePoints.data)

    @Qt.pyqtProperty(ManualFocusScore, notify=manualFocusScoreChanged)
    def manualFocusScore(self):
        return self._manualFocusScore

    @Qt.pyqtProperty(str, notify=experimentDPathChanged)
    def experimentDPath(self):
        return str(self._experimentDPath)

    @experimentDPath.setter
    def experimentDPath(self, experimentDPath):
        experimentDPath = Path(experimentDPath)
        wasValid = self.isValid
        try:
            db = sqlite3.connect(str(experimentDPath / 'analysis' / 'db.sqlite3'))
            db.row_factory = sqlite3.Row
            self._hatchedWellIdxs = SimpleListModel([row['well_idx'] for row in db.execute('select well_idx, did_hatch from wells') if row['did_hatch']])
            self._timePoints = SimpleListModel([row['name'] for row in db.execute('select name from time_points')])
            self._experimentDPath = experimentDPath
            self.db = db
        except:
            self._hatchedWellIdxs = SimpleListModel([])
            self._timePoints = SimpleListModel([])
            self._experimentDPath = None
            raise
        finally:
            self.hatchedWellIdxsChanged.emit()
            self.timePointsChanged.emit()
            isValid = self.isValid
            if wasValid != isValid:
                self.isValidChanged.emit(isValid)
            self.experimentDPathChanged.emit(str(self._experimentDPath))

    @Qt.pyqtProperty(SimpleListModel, notify=hatchedWellIdxsChanged)
    def hatchedWellIdxs(self):
        return self._hatchedWellIdxs

    @Qt.pyqtProperty(SimpleListModel, notify=timePointsChanged)
    def timePoints(self):
        return self._timePoints

    @Qt.pyqtSlot()
    def toPrevTimepoint(self):
        if self.isValid:
            try:
                idx = self._timePoints.data.index(self._manualFocusScore.timepoint)
            except ValueError:
                pass
            else:
                self._manualFocusScore.timepoint = self._timePoints.data[max(idx-1, 0)]

    @Qt.pyqtSlot()
    def toNextTimepoint(self):
        if self.isValid:
            try:
                idx = self._timePoints.data.index(self._manualFocusScore.timepoint)
            except ValueError:
                pass
            else:
                self._manualFocusScore.timepoint = self._timePoints.data[min(idx+1, len(self._timePoints.data)-1)]

    @Qt.pyqtSlot()
    def toPrevWell(self):
        if self.isValid:
            try:
                idx = self._hatchedWellIdxs.data.index(self._manualFocusScore.wellIdx)
            except ValueError:
                pass
            else:
                self._manualFocusScore.wellIdx = self._hatchedWellIdxs.data[max(idx-1, 0)]

    @Qt.pyqtSlot()
    def toNextWell(self):
        if self.isValid:
            try:
                idx = self._hatchedWellIdxs.data.index(self._manualFocusScore.wellIdx)
            except ValueError:
                pass
            else:
                self._manualFocusScore.wellIdx = self._hatchedWellIdxs.data[min(idx+1, len(self._hatchedWellIdxs.data)-1)]

    @Qt.pyqtSlot()
    def uncheckBfFocused(self):
        if self.isValid and self._manualFocusScore.hasBf:
            self._manualFocusScore.bfIsFocused = False

    @Qt.pyqtSlot()
    def checkBfFocused(self):
        if self.isValid and self._manualFocusScore.hasBf:
            self._manualFocusScore.bfIsFocused = True

    @Qt.pyqtSlot()
    def toPrevFocusStackIdx(self):
        if self.isValid and self._manualFocusScore.focusStackLen > 0:
            self._manualFocusScore.bestFocusStackIdx = max(self._manualFocusScore.bestFocusStackIdx-1, -1)

    @Qt.pyqtSlot()
    def toNextFocusStackIdx(self):
        if self.isValid and self._manualFocusScore.focusStackLen > 0:
            self._manualFocusScore.bestFocusStackIdx = min(self._manualFocusScore.bestFocusStackIdx+1, self._manualFocusScore.focusStackLen-1)

def _register_qml_types():
    Qt.qmlRegisterType(SimpleListModel, 'Analysis', 1, 0, 'SimpleListModel')
    Qt.qmlRegisterType(ManualFocusScore, 'Analysis', 1, 0, 'ManualFocusScore')
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
    rw.experiment_manual_focus_scorer.rw = rw
    rw.experiment_manual_focus_scorer._init_actions()
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
    manually_score_focus.experimentDPath = args.experiment_dpath
    rw.show()
    app.exec()
else:
    _register_qml_types()