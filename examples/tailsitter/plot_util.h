#pragma once

#include <vtkAutoInit.h>
#include <vtkAxis.h>
#include <vtkChartLegend.h>
#include <vtkChartMatrix.h>
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkPen.h>
#include <vtkPlot.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkStringArray.h>
#include <vtkTable.h>
#include <vtkTextProperty.h>

#include "drake/common/eigen_types.h"

VTK_AUTOINIT_DECLARE(vtkRenderingOpenGL2)
VTK_AUTOINIT_DECLARE(vtkRenderingContextOpenGL2)
VTK_MODULE_INIT(vtkRenderingFreeType)
VTK_MODULE_INIT(vtkInteractionStyle)

namespace drake {
using std::string;
using std::vector;

namespace internal {
struct vtkOverrides {
  vtkOverrides() {
    VTK_AUTOINIT_CONSTRUCT(vtkRenderingOpenGL2)
    VTK_AUTOINIT_CONSTRUCT(vtkRenderingContextOpenGL2)
  }
};
}  // namespace internal

class Plotter final : private internal::vtkOverrides {
 public:
  Plotter() {}

  void plot(const VectorX<double>& ts, const MatrixX<double>& _x_t,
            vector<string> col_names) {
    const MatrixX<double> x_t{_x_t.topRows(4)};
    const int num_samples = ts.size();
    const int num_dims = x_t.rows();
    const int font_size = 32;

    vtkNew<vtkTable> table;

    vtkNew<vtkFloatArray> t;
    t->SetName("time");
    table->AddColumn(t);
    for (int i = 0; i < num_dims; i++) {
      vtkNew<vtkFloatArray> x_i;
      x_i->SetName(col_names[i].c_str());
      table->AddColumn(x_i);
    }

    // populate data
    table->SetNumberOfRows(num_samples);
    for (int sample = 0; sample < num_samples; sample++) {
      table->SetValue(sample, 0, ts(sample));
      for (int dim = 0; dim < num_dims; dim++) {
        table->SetValue(sample, dim + 1, x_t(dim, sample));
      }
    }

    // add plots
    vtkNew<vtkContextView> view;

    vtkNew<vtkChartMatrix> matrix;
    view->GetScene()->AddItem(matrix);
    matrix->SetSize(vtkVector2i(2, 2));
    matrix->SetGutter(vtkVector2f(200.0, 200.0));
    matrix->SetBorders(100, 100, 100, 100);

    for (int dim = 0; dim < num_dims; dim++) {
      // create chart at appropriate cell in grid
      vtkChart* chart = matrix->GetChart(vtkVector2i(dim / 2, dim % 2));
      chart->SetTitle(fmt::format("{} Plot", col_names[dim]));

      // data to plot
      vtkPlot* plot = chart->AddPlot(vtkChart::LINE);
      plot->SetInputData(table, 0, dim + 1);
      chart->GetAxis(vtkAxis::BOTTOM)->SetTitle(col_names[dim]);
      chart->GetAxis(vtkAxis::LEFT)->SetTitle("time");
      // appearance
      plot->SetColor(0, 255, 0, 255);
      plot->SetWidth(4.0);

      // set font sizes
      chart->GetTitleProperties()->SetFontSize(font_size);
      chart->GetAxis(0)->GetTitleProperties()->SetFontSize(font_size);
      chart->GetAxis(1)->GetTitleProperties()->SetFontSize(font_size);
      plot->GetXAxis()->GetLabelProperties()->SetFontSize(font_size);
      plot->GetYAxis()->GetLabelProperties()->SetFontSize(font_size);
    }

    // render
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    int* size = view->GetRenderWindow()->GetScreenSize();
    const int screen_width = size[0], screen_height = size[1];
    view->GetRenderWindow()->SetSize(screen_width - 400, screen_height - 400);
    view->GetRenderWindow()->SetPosition(200, 200);

    view->GetRenderWindow()->Render();
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
  }
};

}  // namespace drake