// Decorate every code block with a copy-icon button. Blocks containing `$ ` command
// lines copy just the commands (prompts stripped, continuation lines kept); anything
// else (YAML, JSON, score excerpts) copies verbatim.
(function () {
  var ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  var CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';

  function commandText(pre) {
    var lines = pre.innerText.split('\n');
    var out = [];
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].indexOf('$ ') === 0) {
        var cmd = lines[i].slice(2);
        while (/\\\s*$/.test(cmd) && i + 1 < lines.length && lines[i + 1].indexOf('$ ') !== 0) {
          i += 1;
          cmd += '\n' + lines[i];
        }
        out.push(cmd);
      }
    }
    return out.length ? out.join('\n') : pre.innerText;
  }

  document.querySelectorAll('pre').forEach(function (pre) {
    var wrap = document.createElement('div');
    wrap.className = 'pre-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Copy code');
    btn.innerHTML = ICON;
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText(commandText(pre)).then(function () {
        btn.classList.add('done');
        btn.innerHTML = CHECK;
        setTimeout(function () {
          btn.classList.remove('done');
          btn.innerHTML = ICON;
        }, 1500);
      });
    });
    wrap.appendChild(btn);
  });
})();
