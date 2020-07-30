(function () {
    /**
     * Create a cached version of a pure function.
     */
    function cached(fn) {
      var cache = Object.create(null);
      return function (str) {
        var key = isPrimitive(str) ? str : JSON.stringify(str);
        var hit = cache[key];
        return hit || (cache[key] = fn(str))
      }
    }
    
    /**
     * Hyphenate a camelCase string.
     */
    var hyphenate = cached(function (str) {
      return str.replace(/([A-Z])/g, function (m) { return '-' + m.toLowerCase(); })
    });
    
    var hasOwn = Object.prototype.hasOwnProperty;
    
    /**
     * Simple Object.assign polyfill
     */
    var merge =
      Object.assign ||
      function (to) {
        var arguments$1 = arguments;
    
        for (var i = 1; i < arguments.length; i++) {
          var from = Object(arguments$1[i]);
    
          for (var key in from) {
            if (hasOwn.call(from, key)) {
              to[key] = from[key];
            }
          }
        }
    
        return to
      };
    
    /**
     * Check if value is primitive
     */
    function isPrimitive(value) {
      return typeof value === 'string' || typeof value === 'number'
    }
    
    /**
     * Perform no operation.
     */
    function noop() {}
    
    /**
     * Check if value is function
     */
    function isFn(obj) {
      return typeof obj === 'function'
    }
    
    function config () {
      var config = merge(
        {
          el: '#app',
          repo: '',
          maxLevel: 6,
          subMaxLevel: 0,
          loadSidebar: null,
          loadNavbar: null,
          homepage: 'README.md',
          coverpage: '',
          basePath: '',
          auto2top: false,
          name: '',
          themeColor: '',
          nameLink: window.location.pathname,
          autoHeader: false,
          executeScript: null,
          noEmoji: false,
          ga: '',
          ext: '.md',
          mergeNavbar: false,
          formatUpdated: '',
          externalLinkTarget: '_blank',
          routerMode: 'hash',
          noCompileLinks: [],
          relativePath: false
        },
        window.$docsify
      );
    
      var script =
        document.currentScript ||
        [].slice
          .call(document.getElementsByTagName('script'))
          .filter(function (n) { return /docsify\./.test(n.src); })[0];
    
      if (script) {
        for (var prop in config) {
          if (hasOwn.call(config, prop)) {
            var val = script.getAttribute('data-' + hyphenate(prop));
    
            if (isPrimitive(val)) {
              config[prop] = val === '' ? true : val;
            }
          }
        }
    
        if (config.loadSidebar === true) {
          config.loadSidebar = '_sidebar' + config.ext;
        }
        if (config.loadNavbar === true) {
          config.loadNavbar = '_navbar' + config.ext;
        }
        if (config.coverpage === true) {
          config.coverpage = '_coverpage' + config.ext;
        }
        if (config.repo === true) {
          config.repo = '';
        }
        if (config.name === true) {
          config.name = '';
        }
      }
    
      window.$docsify = config;
    
      return config
    }
    
    function initLifecycle(vm) {
      var hooks = [
        'init',
        'mounted',
        'beforeEach',
        'afterEach',
        'doneEach',
        'ready'
      ];
    
      vm._hooks = {};
      vm._lifecycle = {};
      hooks.forEach(function (hook) {
        var arr = (vm._hooks[hook] = []);
        vm._lifecycle[hook] = function (fn) { return arr.push(fn); };
      });
    }
    
    function callHook(vm, hook, data, next) {
      if ( next === void 0 ) next = noop;
    
      var queue = vm._hooks[hook];
    
      var step = function (index) {
        var hook = queue[index];
        if (index >= queue.length) {
          next(data);
        } else if (typeof hook === 'function') {
          if (hook.length === 2) {
            hook(data, function (result) {
              data = result;
              step(index + 1);
            });
          } else {
            var result = hook(data);
            data = result === undefined ? data : result;
            step(index + 1);
          }
        } else {
          step(index + 1);
        }
      };
    
      step(0);
    }
    
    var inBrowser = !false;
    
    var isMobile = inBrowser && document.body.clientWidth <= 600;
    
    /**
     * @see https://github.com/MoOx/pjax/blob/master/lib/is-supported.js
     */
    var supportsPushState =
      inBrowser &&
      (function () {
        // Borrowed wholesale from https://github.com/defunkt/jquery-pjax
        return (
          window.history &&
          window.history.pushState &&
          window.history.replaceState &&
          // PushState isn’t reliable on iOS until 5.
          !navigator.userAgent.match(
            /((iPod|iPhone|iPad).+\bOS\s+[1-4]\D|WebApps\/.+CFNetwork)/
          )
        )
      })();
    
    var cacheNode = {};
    
    /**
     * Get Node
     * @param  {String|Element} el
     * @param  {Boolean} noCache
     * @return {Element}
     */
    function getNode(el, noCache) {
      if ( noCache === void 0 ) noCache = false;
    
      if (typeof el === 'string') {
        if (typeof window.Vue !== 'undefined') {
          return find(el)
        }
        el = noCache ? find(el) : cacheNode[el] || (cacheNode[el] = find(el));
      }
    
      return el
    }
    
    var $ = inBrowser && document;
    
    var body = inBrowser && $.body;
    
    var head = inBrowser && $.head;
    
    /**
     * Find element
     * @example
     * find('nav') => document.querySelector('nav')
     * find(nav, 'a') => nav.querySelector('a')
     */
    function find(el, node) {
      return node ? el.querySelector(node) : $.querySelector(el)
    }
    
    /**
     * Find all elements
     * @example
     * findAll('a') => [].slice.call(document.querySelectorAll('a'))
     * findAll(nav, 'a') => [].slice.call(nav.querySelectorAll('a'))
     */
    function findAll(el, node) {
      return [].slice.call(
        node ? el.querySelectorAll(node) : $.querySelectorAll(el)
      )
    }
    
    function create(node, tpl) {
      node = $.createElement(node);
      if (tpl) {
        node.innerHTML = tpl;
      }
      return node
    }
    
    function appendTo(target, el) {
      return target.appendChild(el)
    }
    
    function before(target, el) {
      return target.insertBefore(el, target.children[0])
    }
    
    function on(el, type, handler) {
      isFn(type) ?
        window.addEventListener(el, type) :
        el.addEventListener(type, handler);
    }
    
    function off(el, type, handler) {
      isFn(type) ?
        window.removeEventListener(el, type) :
        el.removeEventListener(type, handler);
    }
    
    /**
     * Toggle class
     *
     * @example
     * toggleClass(el, 'active') => el.classList.toggle('active')
     * toggleClass(el, 'add', 'active') => el.classList.add('active')
     */
    function toggleClass(el, type, val) {
      el && el.classList[val ? type : 'toggle'](val || type);
    }
    
    function style(content) {
      appendTo(head, create('style', content));
    }
    
    
    var dom = Object.freeze({
        getNode: getNode,
        $: $,
        body: body,
        head: head,
        find: find,
        findAll: findAll,
        create: create,
        appendTo: appendTo,
        before: before,
        on: on,
        off: off,
        toggleClass: toggleClass,
        style: style
    });
    
    /**
     * Render github corner
     * @param  {Object} data
     * @return {String}
     */
    function corner(data) {
      if (!data) {
        return ''
      }
      if (!/\/\//.test(data)) {
        data = 'https://github.com/' + data;
      }
      data = data.replace(/^git\+/, '');
    
      return (
        "<a href=\"" + data + "\" class=\"github-corner\" aria-label=\"View source on Github\">" +
        '<svg viewBox="0 0 250 250" aria-hidden="true">' +
        '<path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>' +
        '<path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>' +
        '<path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>' +
        '</svg>' +
        '</a>'
      )
    }
    
    /**
     * Render main content
     */
    function main(config) {
      var aside =
        '<button class="sidebar-toggle">' +
        '<div class="sidebar-toggle-button">' +
        '<span></span><span></span><span></span>' +
        '</div>' +
        '</button>' +
        '<aside class="sidebar">' +
        (config.name ?
          ("<h1 class=\"app-name\"><a class=\"app-name-link\" data-nosearch>" + (config.logo ?
              ("<img alt=" + (config.name) + " src=" + (config.logo) + ">") :
              config.name) + "</a></h1>") :
          '') +
        '<div class="sidebar-nav"><!--sidebar--></div>' +
        '</aside>';
    
      return (
        (isMobile ? (aside + "<main>") : ("<main>" + aside)) +
        '<section class="content">' +
        '<article class="markdown-section" id="main"><!--main--></article>' +
        '</section>' +
        '</main>'
      )
    }
    
    /**
     * Cover Page
     */
    function cover() {
      var SL = ', 100%, 85%';
      var bgc =
        'linear-gradient(to left bottom, ' +
        "hsl(" + (Math.floor(Math.random() * 255) + SL) + ") 0%," +
        "hsl(" + (Math.floor(Math.random() * 255) + SL) + ") 100%)";
    
      return (
        "<section class=\"cover show\" style=\"background: " + bgc + "\">" +
        '<div class="cover-main"><!--cover--></div>' +
        '<div class="mask"></div>' +
        '</section>'
      )
    }
    
    /**
     * Render tree
     * @param  {Array} tree
     * @param  {String} tpl
     * @return {String}
     */
    function tree(toc, tpl) {
      if ( tpl === void 0 ) tpl = '<ul class="app-sub-sidebar">{inner}</ul>';
    
      if (!toc || !toc.length) {
        return ''
      }
      var innerHTML = '';
      toc.forEach(function (node) {
        innerHTML += "<li><a class=\"section-link\" href=\"" + (node.slug) + "\">" + (node.title) + "</a></li>";
        if (node.children) {
          innerHTML += tree(node.children, tpl);
        }
      });
      return tpl.replace('{inner}', innerHTML)
    }
    
    function helper(className, content) {
      return ("<p class=\"" + className + "\">" + (content.slice(5).trim()) + "</p>")
    }
    
    function theme(color) {
      return ("<style>:root{--theme-color: " + color + ";}</style>")
    }
    
    var barEl;
    var timeId;
    
    /**
     * Init progress component
     */
    function init() {
      var div = create('div');
    
      div.classList.add('progress');
      appendTo(body, div);
      barEl = div;
    }
    /**
     * Render progress bar
     */
    function progressbar (ref) {
      var loaded = ref.loaded;
      var total = ref.total;
      var step = ref.step;
    
      var num;
    
      !barEl && init();
    
      if (step) {
        num = parseInt(barEl.style.width || 0, 10) + step;
        num = num > 80 ? 80 : num;
      } else {
        num = Math.floor(loaded / total * 100);
      }
    
      barEl.style.opacity = 1;
      barEl.style.width = num >= 95 ? '100%' : num + '%';
    
      if (num >= 95) {
        clearTimeout(timeId);
        timeId = setTimeout(function (_) {
          barEl.style.opacity = 0;
          barEl.style.width = '0%';
        }, 200);
      }
    }
    
    var cache = {};
    
    /**
     * Simple ajax get
     * @param {string} url
     * @param {boolean} [hasBar=false] has progress bar
     * @return { then(resolve, reject), abort }
     */
    function get(url, hasBar, headers) {
      if ( hasBar === void 0 ) hasBar = false;
      if ( headers === void 0 ) headers = {};
    
      var xhr = new XMLHttpRequest();
      var on = function () {
        xhr.addEventListener.apply(xhr, arguments);
      };
      var cached$$1 = cache[url];
    
      if (cached$$1) {
        return {then: function (cb) { return cb(cached$$1.content, cached$$1.opt); }, abort: noop}
      }
    
      xhr.open('GET', url);
      for (var i in headers) {
        if (hasOwn.call(headers, i)) {
          xhr.setRequestHeader(i, headers[i]);
        }
      }
      xhr.send();
    
      return {
        then: function (success, error) {
          if ( error === void 0 ) error = noop;
    
          if (hasBar) {
            var id = setInterval(
              function (_) { return progressbar({
                  step: Math.floor(Math.random() * 5 + 1)
                }); },
              500
            );
    
            on('progress', progressbar);
            on('loadend', function (evt) {
              progressbar(evt);
              clearInterval(id);
            });
          }
    
          on('error', error);
          on('load', function (ref) {
            var target = ref.target;
    
            if (target.status >= 400) {
              error(target);
            } else {
              var result = (cache[url] = {
                content: target.response,
                opt: {
                  updatedAt: xhr.getResponseHeader('last-modified')
                }
              });
    
              success(result.content, result.opt);
            }
          });
        },
        abort: function (_) { return xhr.readyState !== 4 && xhr.abort(); }
      }
    }
    
    function replaceVar(block, color) {
      block.innerHTML = block.innerHTML.replace(
        /var\(\s*--theme-color.*?\)/g,
        color
      );
    }
    
    function cssVars (color) {
      // Variable support
      if (window.CSS && window.CSS.supports && window.CSS.supports('(--v:red)')) {
        return
      }
    
      var styleBlocks = findAll('style:not(.inserted),link');
      [].forEach.call(styleBlocks, function (block) {
        if (block.nodeName === 'STYLE') {
          replaceVar(block, color);
        } else if (block.nodeName === 'LINK') {
          var href = block.getAttribute('href');
    
          if (!/\.css$/.test(href)) {
            return
          }
    
          get(href).then(function (res) {
            var style$$1 = create('style', res);
    
            head.appendChild(style$$1);
            replaceVar(style$$1, color);
          });
        }
      });
    }
    
    var RGX = /([^{]*?)\w(?=\})/g;
    
    var dict = {
        YYYY: 'getFullYear',
        YY: 'getYear',
        MM: function (d) {
            return d.getMonth() + 1;
        },
        DD: 'getDate',
        HH: 'getHours',
        mm: 'getMinutes',
        ss: 'getSeconds'
    };
    
    function tinydate (str) {
        var parts=[], offset=0;
        str.replace(RGX, function (key, _, idx) {
            // save preceding string
            parts.push(str.substring(offset, idx - 1));
            offset = idx += key.length + 1;
            // save function
            parts.push(function(d){
                return ('00' + (typeof dict[key]==='string' ? d[dict[key]]() : dict[key](d))).slice(-key.length);
            });
        });
    
        if (offset !== str.length) {
            parts.push(str.substring(offset));
        }
    
        return function (arg) {
            var out='', i=0, d=arg||new Date();
            for (; i<parts.length; i++) {
                out += (typeof parts[i]==='string') ? parts[i] : parts[i](d);
            }
            return out;
        };
    }
    
    var commonjsGlobal = typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};
    
    
    
    
    
    function createCommonjsModule(fn, module) {
        return module = { exports: {} }, fn(module, module.exports), module.exports;
    }
    
    var marked = createCommonjsModule(function (module, exports) {
    /**
     * marked - a markdown parser
     * Copyright (c) 2011-2018, Christopher Jeffrey. (MIT Licensed)
     * https://github.com/markedjs/marked
     */
    
    (function(root) {
    var block = {
      newline: /^\n+/,
      code: /^( {4}[^\n]+\n*)+/,
      fences: noop,
      hr: /^ {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)/,
      heading: /^ *(#{1,6}) *([^\n]+?) *(?:#+ *)?(?:\n+|$)/,
      nptable: noop,
      blockquote: /^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/,
      blocklatex: /^(\${2}) *((?:[^$]|\\\$)+?) *\n\1(?:\n|$)/,
      list: /^( *)(bull) [\s\S]+?(?:hr|def|\n{2,}(?! )(?!\1bull )\n*|\s*$)/,
      html: '^ {0,3}(?:' // optional indentation
        + '<(script|pre|style)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)' // (1)
        + '|comment[^\\n]*(\\n+|$)' // (2)
        + '|<\\?[\\s\\S]*?\\?>\\n*' // (3)
        + '|<![A-Z][\\s\\S]*?>\\n*' // (4)
        + '|<!\\[CDATA\\[[\\s\\S]*?\\]\\]>\\n*' // (5)
        + '|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:\\n{2,}|$)' // (6)
        + '|<(?!script|pre|style)([a-z][\\w-]*)(?:attribute)*? */?>(?=\\h*\\n)[\\s\\S]*?(?:\\n{2,}|$)' // (7) open tag
        + '|</(?!script|pre|style)[a-z][\\w-]*\\s*>(?=\\h*\\n)[\\s\\S]*?(?:\\n{2,}|$)' // (7) closing tag
        + ')',
      def: /^ {0,3}\[(label)\]: *\n? *<?([^\s>]+)>?(?:(?: +\n? *| *\n *)(title))? *(?:\n+|$)/,
      table: noop,
      lheading: /^([^\n]+)\n *(=|-){2,} *(?:\n+|$)/,
      paragraph: /^([^\n]+(?:\n(?!hr|heading|lheading| {0,3}>|<\/?(?:tag)(?: +|\n|\/?>)|<(?:script|pre|style|!--))[^\n]+)*)/,
      text: /^[^\n]+/
    };
    
    block._label = /(?!\s*\])(?:\\[\[\]]|[^\[\]])+/;
    block._title = /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/;
    block.def = edit(block.def)
      .replace('label', block._label)
      .replace('title', block._title)
      .getRegex();
    
    block.bullet = /(?:[*+-]|\d+\.)/;
    block.item = /^( *)(bull) [^\n]*(?:\n(?!\1bull )[^\n]*)*/;
    block.item = edit(block.item, 'gm')
      .replace(/bull/g, block.bullet)
      .getRegex();
    
    block.list = edit(block.list)
      .replace(/bull/g, block.bullet)
      .replace('hr', '\\n+(?=\\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$))')
      .replace('def', '\\n+(?=' + block.def.source + ')')
      .getRegex();
    
    block._tag = 'address|article|aside|base|basefont|blockquote|body|caption'
      + '|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption'
      + '|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe'
      + '|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option'
      + '|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr'
      + '|track|ul';
    block._comment = /<!--(?!-?>)[\s\S]*?-->/;
    block.html = edit(block.html, 'i')
      .replace('comment', block._comment)
      .replace('tag', block._tag)
      .replace('attribute', / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/)
      .getRegex();
    
    block.paragraph = edit(block.paragraph)
      .replace('hr', block.hr)
      .replace('heading', block.heading)
      .replace('lheading', block.lheading)
      .replace('tag', block._tag) // pars can be interrupted by type (6) html blocks
      .getRegex();
    
    block.blockquote = edit(block.blockquote)
      .replace('paragraph', block.paragraph)
      .getRegex();
    
    /**
     * Normal Block Grammar
     */
    
    block.normal = merge({}, block);
    
    /**
     * GFM Block Grammar
     */
    
    block.gfm = merge({}, block.normal, {
      fences: /^ *(`{3,}|~{3,})[ \.]*(\S+)? *\n([\s\S]*?)\n? *\1 *(?:\n+|$)/,
      paragraph: /^/,
      heading: /^ *(#{1,6}) +([^\n]+?) *#* *(?:\n+|$)/
    });
    
    block.gfm.paragraph = edit(block.paragraph)
      .replace('(?!', '(?!'
        + block.gfm.fences.source.replace('\\1', '\\2') + '|'
        + block.list.source.replace('\\1', '\\3') + '|')
      .getRegex();
    
    /**
     * GFM + Tables Block Grammar
     */
    
    block.tables = merge({}, block.gfm, {
      nptable: /^ *([^|\n ].*\|.*)\n *([-:]+ *\|[-| :]*)(?:\n((?:.*[^>\n ].*(?:\n|$))*)\n*|$)/,
      table: /^ *\|(.+)\n *\|?( *[-:]+[-| :]*)(?:\n((?: *[^>\n ].*(?:\n|$))*)\n*|$)/
    });
    
    /**
     * Pedantic grammar
     */
    
    block.pedantic = merge({}, block.normal, {
      html: edit(
        '^ *(?:comment *(?:\\n|\\s*$)'
        + '|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)' // closed tag
        + '|<tag(?:"[^"]*"|\'[^\']*\'|\\s[^\'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))')
        .replace('comment', block._comment)
        .replace(/tag/g, '(?!(?:'
          + 'a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub'
          + '|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)'
          + '\\b)\\w+(?!:|[^\\w\\s@]*@)\\b')
        .getRegex(),
      def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/
    });
    
    /**
     * Block Lexer
     */
    
    function Lexer(options) {
      this.tokens = [];
      this.tokens.links = {};
      this.options = options || marked.defaults;
      this.rules = block.normal;
    
      if (this.options.pedantic) {
        this.rules = block.pedantic;
      } else if (this.options.gfm) {
        if (this.options.tables) {
          this.rules = block.tables;
        } else {
          this.rules = block.gfm;
        }
      }
    }
    
    /**
     * Expose Block Rules
     */
    
    Lexer.rules = block;
    
    /**
     * Static Lex Method
     */
    
    Lexer.lex = function(src, options) {
      var lexer = new Lexer(options);
      return lexer.lex(src);
    };
    
    /**
     * Preprocessing
     */
    
    Lexer.prototype.lex = function(src) {
      src = src
        .replace(/\r\n|\r/g, '\n')
        .replace(/\t/g, '    ')
        .replace(/\u00a0/g, ' ')
        .replace(/\u2424/g, '\n');
    
      return this.token(src, true);
    };
    
    /**
     * Lexing
     */
    
    Lexer.prototype.token = function(src, top) {
      src = src.replace(/^ +$/gm, '');
      var next,
          loose,
          cap,
          bull,
          b,
          item,
          space,
          i,
          tag,
          l,
          isordered,
          istask,
          ischecked;
    
      while (src) {
        // newline
        if (cap = this.rules.newline.exec(src)) {
          src = src.substring(cap[0].length);
          if (cap[0].length > 1) {
            this.tokens.push({
              type: 'space'
            });
          }
        }
    
        // code
        if (cap = this.rules.code.exec(src)) {
          src = src.substring(cap[0].length);
          cap = cap[0].replace(/^ {4}/gm, '');
          this.tokens.push({
            type: 'code',
            text: !this.options.pedantic
              ? cap.replace(/\n+$/, '')
              : cap
          });
          continue;
        }
    
        // fences (gfm)
        if (cap = this.rules.fences.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'code',
            lang: cap[2],
            text: cap[3] || ''
          });
          continue;
        }
    
        // heading
        if (cap = this.rules.heading.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'heading',
            depth: cap[1].length,
            text: cap[2]
          });
          continue;
        }
    
        // latex
        if (cap = this.rules.blocklatex.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'blocklatex',
            text: cap[2]
          });
          continue;
        }
    
        // table no leading pipe (gfm)
        if (top && (cap = this.rules.nptable.exec(src))) {
          item = {
            type: 'table',
            header: splitCells(cap[1].replace(/^ *| *\| *$/g, '')),
            align: cap[2].replace(/^ *|\| *$/g, '').split(/ *\| */),
            cells: cap[3] ? cap[3].replace(/\n$/, '').split('\n') : []
          };
    
          if (item.header.length === item.align.length) {
            src = src.substring(cap[0].length);
    
            for (i = 0; i < item.align.length; i++) {
              if (/^ *-+: *$/.test(item.align[i])) {
                item.align[i] = 'right';
              } else if (/^ *:-+: *$/.test(item.align[i])) {
                item.align[i] = 'center';
              } else if (/^ *:-+ *$/.test(item.align[i])) {
                item.align[i] = 'left';
              } else {
                item.align[i] = null;
              }
            }
    
            for (i = 0; i < item.cells.length; i++) {
              item.cells[i] = splitCells(item.cells[i], item.header.length);
            }
    
            this.tokens.push(item);
    
            continue;
          }
        }
    
        // hr
        if (cap = this.rules.hr.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'hr'
          });
          continue;
        }
    
        // blockquote
        if (cap = this.rules.blockquote.exec(src)) {
          src = src.substring(cap[0].length);
    
          this.tokens.push({
            type: 'blockquote_start'
          });
    
          cap = cap[0].replace(/^ *> ?/gm, '');
    
          // Pass `top` to keep the current
          // "toplevel" state. This is exactly
          // how markdown.pl works.
          this.token(cap, top);
    
          this.tokens.push({
            type: 'blockquote_end'
          });
    
          continue;
        }
    
        // list
        if (cap = this.rules.list.exec(src)) {
          src = src.substring(cap[0].length);
          bull = cap[2];
          isordered = bull.length > 1;
    
          this.tokens.push({
            type: 'list_start',
            ordered: isordered,
            start: isordered ? +bull : ''
          });
    
          // Get each top-level item.
          cap = cap[0].match(this.rules.item);
    
          next = false;
          l = cap.length;
          i = 0;
    
          for (; i < l; i++) {
            item = cap[i];
    
            // Remove the list item's bullet
            // so it is seen as the next token.
            space = item.length;
            item = item.replace(/^ *([*+-]|\d+\.) +/, '');
    
            // Outdent whatever the
            // list item contains. Hacky.
            if (~item.indexOf('\n ')) {
              space -= item.length;
              item = !this.options.pedantic
                ? item.replace(new RegExp('^ {1,' + space + '}', 'gm'), '')
                : item.replace(/^ {1,4}/gm, '');
            }
    
            // Determine whether the next list item belongs here.
            // Backpedal if it does not belong in this list.
            if (this.options.smartLists && i !== l - 1) {
              b = block.bullet.exec(cap[i + 1])[0];
              if (bull !== b && !(bull.length > 1 && b.length > 1)) {
                src = cap.slice(i + 1).join('\n') + src;
                i = l - 1;
              }
            }
    
            // Determine whether item is loose or not.
            // Use: /(^|\n)(?! )[^\n]+\n\n(?!\s*$)/
            // for discount behavior.
            loose = next || /\n\n(?!\s*$)/.test(item);
            if (i !== l - 1) {
              next = item.charAt(item.length - 1) === '\n';
              if (!loose) loose = next;
            }
    
            // Check for task list items
            istask = /^\[[ xX]\] /.test(item);
            ischecked = undefined;
            if (istask) {
              ischecked = item[1] !== ' ';
              item = item.replace(/^\[[ xX]\] +/, '');
            }
    
            this.tokens.push({
              type: loose
                ? 'loose_item_start'
                : 'list_item_start',
              task: istask,
              checked: ischecked
            });
    
            // Recurse.
            this.token(item, false);
    
            this.tokens.push({
              type: 'list_item_end'
            });
          }
    
          this.tokens.push({
            type: 'list_end'
          });
    
          continue;
        }
    
        // html
        if (cap = this.rules.html.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: this.options.sanitize
              ? 'paragraph'
              : 'html',
            pre: !this.options.sanitizer
              && (cap[1] === 'pre' || cap[1] === 'script' || cap[1] === 'style'),
            text: cap[0]
          });
          continue;
        }
    
        // def
        if (top && (cap = this.rules.def.exec(src))) {
          src = src.substring(cap[0].length);
          if (cap[3]) cap[3] = cap[3].substring(1, cap[3].length - 1);
          tag = cap[1].toLowerCase().replace(/\s+/g, ' ');
          if (!this.tokens.links[tag]) {
            this.tokens.links[tag] = {
              href: cap[2],
              title: cap[3]
            };
          }
          continue;
        }
    
        // table (gfm)
        if (top && (cap = this.rules.table.exec(src))) {
          item = {
            type: 'table',
            header: splitCells(cap[1].replace(/^ *| *\| *$/g, '')),
            align: cap[2].replace(/^ *|\| *$/g, '').split(/ *\| */),
            cells: cap[3] ? cap[3].replace(/(?: *\| *)?\n$/, '').split('\n') : []
          };
    
          if (item.header.length === item.align.length) {
            src = src.substring(cap[0].length);
    
            for (i = 0; i < item.align.length; i++) {
              if (/^ *-+: *$/.test(item.align[i])) {
                item.align[i] = 'right';
              } else if (/^ *:-+: *$/.test(item.align[i])) {
                item.align[i] = 'center';
              } else if (/^ *:-+ *$/.test(item.align[i])) {
                item.align[i] = 'left';
              } else {
                item.align[i] = null;
              }
            }
    
            for (i = 0; i < item.cells.length; i++) {
              item.cells[i] = splitCells(
                item.cells[i].replace(/^ *\| *| *\| *$/g, ''),
                item.header.length);
            }
    
            this.tokens.push(item);
    
            continue;
          }
        }
    
        // lheading
        if (cap = this.rules.lheading.exec(src)) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'heading',
            depth: cap[2] === '=' ? 1 : 2,
            text: cap[1]
          });
          continue;
        }
    
        // top-level paragraph
        if (top && (cap = this.rules.paragraph.exec(src))) {
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'paragraph',
            text: cap[1].charAt(cap[1].length - 1) === '\n'
              ? cap[1].slice(0, -1)
              : cap[1]
          });
          continue;
        }
    
        // text
        if (cap = this.rules.text.exec(src)) {
          // Top-level should never reach here.
          src = src.substring(cap[0].length);
          this.tokens.push({
            type: 'text',
            text: cap[0]
          });
          continue;
        }
    
        if (src) {
          throw new Error('Infinite loop on byte: ' + src.charCodeAt(0));
        }
      }
    
      return this.tokens;
    };
    
    /**
     * Inline-Level Grammar
     */
    
    var inline = {
      escape: /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,
      autolink: /^<(scheme:[^\s\x00-\x1f<>]*|email)>/,
      url: noop,
      tag: '^comment'
        + '|^</[a-zA-Z][\\w:-]*\\s*>' // self-closing tag
        + '|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>' // open tag
        + '|^<\\?[\\s\\S]*?\\?>' // processing instruction, e.g. <?php ?>
        + '|^<![a-zA-Z]+\\s[\\s\\S]*?>' // declaration, e.g. <!DOCTYPE html>
        + '|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>', // CDATA section
      link: /^!?\[(label)\]\(href(?:\s+(title))?\s*\)/,
      reflink: /^!?\[(label)\]\[(?!\s*\])((?:\\[\[\]]?|[^\[\]\\])+)\]/,
      nolink: /^!?\[(?!\s*\])((?:\[[^\[\]]*\]|\\[\[\]]|[^\[\]])*)\](?:\[\])?/,
      strong: /^__([^\s][\s\S]*?[^\s])__(?!_)|^\*\*([^\s][\s\S]*?[^\s])\*\*(?!\*)|^__([^\s])__(?!_)|^\*\*([^\s])\*\*(?!\*)/,
      em: /^_([^\s][\s\S]*?[^\s_])_(?!_)|^_([^\s_][\s\S]*?[^\s])_(?!_)|^\*([^\s][\s\S]*?[^\s*])\*(?!\*)|^\*([^\s*][\s\S]*?[^\s])\*(?!\*)|^_([^\s_])_(?!_)|^\*([^\s*])\*(?!\*)/,
      code: /^(`+)\s*([\s\S]*?[^`]?)\s*\1(?!`)/,
      inlinelatex: /^(\${1,2}) *((?:[^\\$]|\\\S)+?) *\1(?!\$)/,
      br: /^ {2,}\n(?!\s*$)/,
      del: noop,
      text: /^[\s\S]+?(?=[\\<!\[`*$]|\b_| {2,}\n|$)/
    };
    
    inline._escapes = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/g;
    
    inline._scheme = /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/;
    inline._email = /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/;
    inline.autolink = edit(inline.autolink)
      .replace('scheme', inline._scheme)
      .replace('email', inline._email)
      .getRegex();
    
    inline._attribute = /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/;
    
    inline.tag = edit(inline.tag)
      .replace('comment', block._comment)
      .replace('attribute', inline._attribute)
      .getRegex();
    
    inline._label = /(?:\[[^\[\]]*\]|\\[\[\]]?|`[^`]*`|[^\[\]\\])*?/;
    inline._href = /\s*(<(?:\\[<>]?|[^\s<>\\])*>|(?:\\[()]?|\([^\s\x00-\x1f()\\]*\)|[^\s\x00-\x1f()\\])*?)/;
    inline._title = /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/;
    
    inline.link = edit(inline.link)
      .replace('label', inline._label)
      .replace('href', inline._href)
      .replace('title', inline._title)
      .getRegex();
    
    inline.reflink = edit(inline.reflink)
      .replace('label', inline._label)
      .getRegex();
    
    /**
     * Normal Inline Grammar
     */
    
    inline.normal = merge({}, inline);
    
    /**
     * Pedantic Inline Grammar
     */
    
    inline.pedantic = merge({}, inline.normal, {
      strong: /^__(?=\S)([\s\S]*?\S)__(?!_)|^\*\*(?=\S)([\s\S]*?\S)\*\*(?!\*)/,
      em: /^_(?=\S)([\s\S]*?\S)_(?!_)|^\*(?=\S)([\s\S]*?\S)\*(?!\*)/,
      link: edit(/^!?\[(label)\]\((.*?)\)/)
        .replace('label', inline._label)
        .getRegex(),
      reflink: edit(/^!?\[(label)\]\s*\[([^\]]*)\]/)
        .replace('label', inline._label)
        .getRegex()
    });
    
    /**
     * GFM Inline Grammar
     */
    
    inline.gfm = merge({}, inline.normal, {
      escape: edit(inline.escape).replace('])', '~|])').getRegex(),
      url: edit(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/)
        .replace('email', inline._email)
        .getRegex(),
      _backpedal: /(?:[^?!.,:;*_~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_~)]+(?!$))+/,
      del: /^~~(?=\S)([\s\S]*?\S)~~/,
      text: edit(inline.text)
        .replace(']|', '~]|')
        .replace('|', '|https?://|ftp://|www\\.|[a-zA-Z0-9.!#$%&\'*+/=?^_`{\\|}~-]+@|')
        .getRegex()
    });
    
    /**
     * GFM + Line Breaks Inline Grammar
     */
    
    inline.breaks = merge({}, inline.gfm, {
      br: edit(inline.br).replace('{2,}', '*').getRegex(),
      text: edit(inline.gfm.text).replace('{2,}', '*').getRegex()
    });
    
    /**
     * Inline Lexer & Compiler
     */
    
    function InlineLexer(links, options) {
      this.options = options || marked.defaults;
      this.links = links;
      this.rules = inline.normal;
      this.renderer = this.options.renderer || new Renderer();
      this.renderer.latex = this.renderer.latex || Renderer.prototype.latex
      this.renderer.options = this.options;
    
      if (!this.links) {
        throw new Error('Tokens array requires a `links` property.');
      }
    
      if (this.options.pedantic) {
        this.rules = inline.pedantic;
      } else if (this.options.gfm) {
        if (this.options.breaks) {
          this.rules = inline.breaks;
        } else {
          this.rules = inline.gfm;
        }
      }
    }
    
    /**
     * Expose Inline Rules
     */
    
    InlineLexer.rules = inline;
    
    /**
     * Static Lexing/Compiling Method
     */
    
    InlineLexer.output = function(src, links, options) {
      var inline = new InlineLexer(links, options);
      return inline.output(src);
    };
    
    /**
     * Lexing/Compiling
     */
    
    InlineLexer.prototype.output = function(src) {
      var out = '',
          link,
          text,
          href,
          title,
          cap;
    
      while (src) {
        // escape
        if (cap = this.rules.escape.exec(src)) {
          src = src.substring(cap[0].length);
          out += cap[1];
          continue;
        }
    
        // autolink
        if (cap = this.rules.autolink.exec(src)) {
          src = src.substring(cap[0].length);
          if (cap[2] === '@') {
            text = escape(this.mangle(cap[1]));
            href = 'mailto:' + text;
          } else {
            text = escape(cap[1]);
            href = text;
          }
          out += this.renderer.link(href, null, text);
          continue;
        }
    
        // url (gfm)
        if (!this.inLink && (cap = this.rules.url.exec(src))) {
          cap[0] = this.rules._backpedal.exec(cap[0])[0];
          src = src.substring(cap[0].length);
          if (cap[2] === '@') {
            text = escape(cap[0]);
            href = 'mailto:' + text;
          } else {
            text = escape(cap[0]);
            if (cap[1] === 'www.') {
              href = 'http://' + text;
            } else {
              href = text;
            }
          }
          out += this.renderer.link(href, null, text);
          continue;
        }
    
        // tag
        if (cap = this.rules.tag.exec(src)) {
          if (!this.inLink && /^<a /i.test(cap[0])) {
            this.inLink = true;
          } else if (this.inLink && /^<\/a>/i.test(cap[0])) {
            this.inLink = false;
          }
          src = src.substring(cap[0].length);
          out += this.options.sanitize
            ? this.options.sanitizer
              ? this.options.sanitizer(cap[0])
              : escape(cap[0])
            : cap[0]
          continue;
        }
    
        // link
        if (cap = this.rules.link.exec(src)) {
          src = src.substring(cap[0].length);
          this.inLink = true;
          href = cap[2];
          if (this.options.pedantic) {
            link = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(href);
    
            if (link) {
              href = link[1];
              title = link[3];
            } else {
              title = '';
            }
          } else {
            title = cap[3] ? cap[3].slice(1, -1) : '';
          }
          href = href.trim().replace(/^<([\s\S]*)>$/, '$1');
          out += this.outputLink(cap, {
            href: InlineLexer.escapes(href),
            title: InlineLexer.escapes(title)
          });
          this.inLink = false;
          continue;
        }
    
        // reflink, nolink
        if ((cap = this.rules.reflink.exec(src))
            || (cap = this.rules.nolink.exec(src))) {
          src = src.substring(cap[0].length);
          link = (cap[2] || cap[1]).replace(/\s+/g, ' ');
          link = this.links[link.toLowerCase()];
          if (!link || !link.href) {
            out += cap[0].charAt(0);
            src = cap[0].substring(1) + src;
            continue;
          }
          this.inLink = true;
          out += this.outputLink(cap, link);
          this.inLink = false;
          continue;
        }
    
        // strong
        if (cap = this.rules.strong.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.strong(this.output(cap[4] || cap[3] || cap[2] || cap[1]));
          continue;
        }
    
        // em
        if (cap = this.rules.em.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.em(this.output(cap[6] || cap[5] || cap[4] || cap[3] || cap[2] || cap[1]));
          continue;
        }
    
        // code
        if (cap = this.rules.code.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.codespan(escape(cap[2].trim(), true));
          continue;
        }
    
        // latex
        if (cap = this.rules.inlinelatex.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.latex(cap[2], '$$'===cap[1])
          continue;
        }
    
        // br
        if (cap = this.rules.br.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.br();
          continue;
        }
    
        // del (gfm)
        if (cap = this.rules.del.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.del(this.output(cap[1]));
          continue;
        }
    
        // text
        if (cap = this.rules.text.exec(src)) {
          src = src.substring(cap[0].length);
          out += this.renderer.text(escape(this.smartypants(cap[0])));
          continue;
        }
    
        if (src) {
          throw new Error('Infinite loop on byte: ' + src.charCodeAt(0));
        }
      }
    
      return out;
    };
    
    InlineLexer.escapes = function(text) {
      return text ? text.replace(InlineLexer.rules._escapes, '$1') : text;
    }
    
    /**
     * Compile Link
     */
    
    InlineLexer.prototype.outputLink = function(cap, link) {
      var href = link.href,
          title = link.title ? escape(link.title) : null;
    
      return cap[0].charAt(0) !== '!'
        ? this.renderer.link(href, title, this.output(cap[1]))
        : this.renderer.image(href, title, escape(cap[1]));
    };
    
    /**
     * Smartypants Transformations
     */
    
    InlineLexer.prototype.smartypants = function(text) {
      if (!this.options.smartypants) return text;
      return text
        // em-dashes
        .replace(/---/g, '\u2014')
        // en-dashes
        .replace(/--/g, '\u2013')
        // opening singles
        .replace(/(^|[-\u2014/(\[{"\s])'/g, '$1\u2018')
        // closing singles & apostrophes
        .replace(/'/g, '\u2019')
        // opening doubles
        .replace(/(^|[-\u2014/(\[{\u2018\s])"/g, '$1\u201c')
        // closing doubles
        .replace(/"/g, '\u201d')
        // ellipses
        .replace(/\.{3}/g, '\u2026');
    };
    
    /**
     * Mangle Links
     */
    
    InlineLexer.prototype.mangle = function(text) {
      if (!this.options.mangle) return text;
      var out = '',
          l = text.length,
          i = 0,
          ch;
    
      for (; i < l; i++) {
        ch = text.charCodeAt(i);
        if (Math.random() > 0.5) {
          ch = 'x' + ch.toString(16);
        }
        out += '&#' + ch + ';';
      }
    
      return out;
    };
    
    /**
     * Renderer
     */
    
    function Renderer(options) {
      this.options = options || marked.defaults;
    }
    
    Renderer.prototype.code = function(code, lang, escaped) {
      if (this.options.highlight) {
        var out = this.options.highlight(code, lang);
        if (out != null && out !== code) {
          escaped = true;
          code = out;
        }
      }
    
      if (!lang) {
        return '<pre><code>'
          + (escaped ? code : escape(code, true))
          + '</code></pre>';
      }
    
      return '<pre><code class="'
        + this.options.langPrefix
        + escape(lang, true)
        + '">'
        + (escaped ? code : escape(code, true))
        + '</code></pre>\n';
    };
    
    Renderer.prototype.blockquote = function(quote) {
      return '<blockquote>\n' + quote + '</blockquote>\n';
    };
    
    Renderer.prototype.html = function(html) {
      return html;
    };
    
    Renderer.prototype.latex = function(text, block=false) {
      var out;
      try {
        if (marked.defaults.latexRender) {
          var html = marked.defaults.latexRender(text);
          if (block) {
            out = '<div style="text-align:center; maring:10px;padding:10px;">' + html + '</div>';
          }else{
            out = html;
          }
        } else {
          console.log('No latexRender');
        }
      } catch (e) {
        console.info('Failed to render latex: "' + text + '"');
        out = '$$' + escape(text) + '$$';
      }
      return out;
    };
    
    Renderer.prototype.heading = function(text, level, raw) {
      if (this.options.headerIds) {
        return '<h'
          + level
          + ' id="'
          + this.options.headerPrefix
          + raw.toLowerCase().replace(/[^\w]+/g, '-')
          + '">'
          + text
          + '</h'
          + level
          + '>\n';
      }
      // ignore IDs
      return '<h' + level + '>' + text + '</h' + level + '>\n';
    };
    
    Renderer.prototype.hr = function() {
      return this.options.xhtml ? '<hr/>\n' : '<hr>\n';
    };
    
    Renderer.prototype.list = function(body, ordered, start) {
      var type = ordered ? 'ol' : 'ul',
          startatt = (ordered && start !== 1) ? (' start="' + start + '"') : '';
      return '<' + type + startatt + '>\n' + body + '</' + type + '>\n';
    };
    
    Renderer.prototype.listitem = function(text) {
      return '<li>' + text + '</li>\n';
    };
    
    Renderer.prototype.checkbox = function(checked) {
      return '<input '
        + (checked ? 'checked="" ' : '')
        + 'disabled="" type="checkbox"'
        + (this.options.xhtml ? ' /' : '')
        + '> ';
    }
    
    Renderer.prototype.paragraph = function(text) {
      return '<p>' + text + '</p>\n';
    };
    
    Renderer.prototype.table = function(header, body) {
      if (body) body = '<tbody>' + body + '</tbody>';
    
      return '<table>\n'
        + '<thead>\n'
        + header
        + '</thead>\n'
        + body
        + '</table>\n';
    };
    
    Renderer.prototype.tablerow = function(content) {
      return '<tr>\n' + content + '</tr>\n';
    };
    
    Renderer.prototype.tablecell = function(content, flags) {
      var type = flags.header ? 'th' : 'td';
      var tag = flags.align
        ? '<' + type + ' align="' + flags.align + '">'
        : '<' + type + '>';
      return tag + content + '</' + type + '>\n';
    };
    
    // span level renderer
    Renderer.prototype.strong = function(text) {
      return '<strong>' + text + '</strong>';
    };
    
    Renderer.prototype.em = function(text) {
      return '<em>' + text + '</em>';
    };
    
    Renderer.prototype.codespan = function(text) {
      return '<code>' + text + '</code>';
    };
    
    Renderer.prototype.br = function() {
      return this.options.xhtml ? '<br/>' : '<br>';
    };
    
    Renderer.prototype.del = function(text) {
      return '<del>' + text + '</del>';
    };
    
    Renderer.prototype.link = function(href, title, text) {
      if (this.options.sanitize) {
        try {
          var prot = decodeURIComponent(unescape(href))
            .replace(/[^\w:]/g, '')
            .toLowerCase();
        } catch (e) {
          return text;
        }
        if (prot.indexOf('javascript:') === 0 || prot.indexOf('vbscript:') === 0 || prot.indexOf('data:') === 0) {
          return text;
        }
      }
      if (this.options.baseUrl && !originIndependentUrl.test(href)) {
        href = resolveUrl(this.options.baseUrl, href);
      }
      try {
        href = encodeURI(href).replace(/%25/g, '%');
      } catch (e) {
        return text;
      }
      var out = '<a href="' + escape(href) + '"';
      if (title) {
        out += ' title="' + title + '"';
      }
      out += '>' + text + '</a>';
      return out;
    };
    
    Renderer.prototype.image = function(href, title, text) {
      if (this.options.baseUrl && !originIndependentUrl.test(href)) {
        href = resolveUrl(this.options.baseUrl, href);
      }
      var out = '<img src="' + href + '" alt="' + text + '"';
      if (title) {
        out += ' title="' + title + '"';
      }
      out += this.options.xhtml ? '/>' : '>';
      return out;
    };
    
    Renderer.prototype.text = function(text) {
      return text;
    };
    
    /**
     * TextRenderer
     * returns only the textual part of the token
     */
    
    function TextRenderer() {}
    
    // no need for block level renderers
    
    TextRenderer.prototype.strong =
    TextRenderer.prototype.em =
    TextRenderer.prototype.codespan =
    TextRenderer.prototype.del =
    TextRenderer.prototype.text = function (text) {
      return text;
    }
    
    TextRenderer.prototype.link =
    TextRenderer.prototype.image = function(href, title, text) {
      return '' + text;
    }
    
    TextRenderer.prototype.br = function() {
      return '';
    }
    
    /**
     * Parsing & Compiling
     */
    
    function Parser(options) {
      this.tokens = [];
      this.token = null;
      this.options = options || marked.defaults;
      this.options.renderer = this.options.renderer || new Renderer();
      this.renderer = this.options.renderer;
      this.renderer.options = this.options;
    }
    
    /**
     * Static Parse Method
     */
    
    Parser.parse = function(src, options) {
      var parser = new Parser(options);
      return parser.parse(src);
    };
    
    /**
     * Parse Loop
     */
    
    Parser.prototype.parse = function(src) {
      this.inline = new InlineLexer(src.links, this.options);
      // use an InlineLexer with a TextRenderer to extract pure text
      this.inlineText = new InlineLexer(
        src.links,
        merge({}, this.options, {renderer: new TextRenderer()})
      );
      this.tokens = src.reverse();
    
      var out = '';
      while (this.next()) {
        out += this.tok();
      }
    
      return out;
    };
    
    /**
     * Next Token
     */
    
    Parser.prototype.next = function() {
      return this.token = this.tokens.pop();
    };
    
    /**
     * Preview Next Token
     */
    
    Parser.prototype.peek = function() {
      return this.tokens[this.tokens.length - 1] || 0;
    };
    
    /**
     * Parse Text Tokens
     */
    
    Parser.prototype.parseText = function() {
      var body = this.token.text;
    
      while (this.peek().type === 'text') {
        body += '\n' + this.next().text;
      }
    
      return this.inline.output(body);
    };
    
    /**
     * Parse Current Token
     */
    
    Parser.prototype.tok = function() {
      switch (this.token.type) {
        case 'space': {
          return '';
        }
        case 'hr': {
          return this.renderer.hr();
        }
        case 'heading': {
          return this.renderer.heading(
            this.inline.output(this.token.text),
            this.token.depth,
            unescape(this.inlineText.output(this.token.text)));
        }
        case 'blocklatex': {
          return this.renderer.latex(this.token.text, true)
        }
        case 'code': {
          return this.renderer.code(this.token.text,
            this.token.lang,
            this.token.escaped);
        }
        case 'table': {
          var header = '',
              body = '',
              i,
              row,
              cell,
              j;
    
          // header
          cell = '';
          for (i = 0; i < this.token.header.length; i++) {
            cell += this.renderer.tablecell(
              this.inline.output(this.token.header[i]),
              { header: true, align: this.token.align[i] }
            );
          }
          header += this.renderer.tablerow(cell);
    
          for (i = 0; i < this.token.cells.length; i++) {
            row = this.token.cells[i];
    
            cell = '';
            for (j = 0; j < row.length; j++) {
              cell += this.renderer.tablecell(
                this.inline.output(row[j]),
                { header: false, align: this.token.align[j] }
              );
            }
    
            body += this.renderer.tablerow(cell);
          }
          return this.renderer.table(header, body);
        }
        case 'blockquote_start': {
          body = '';
    
          while (this.next().type !== 'blockquote_end') {
            body += this.tok();
          }
    
          return this.renderer.blockquote(body);
        }
        case 'list_start': {
          body = '';
          var ordered = this.token.ordered,
              start = this.token.start;
    
          while (this.next().type !== 'list_end') {
            body += this.tok();
          }
    
          return this.renderer.list(body, ordered, start);
        }
        case 'list_item_start': {
          body = '';
    
          if (this.token.task) {
            body += this.renderer.checkbox(this.token.checked);
          }
    
          while (this.next().type !== 'list_item_end') {
            body += this.token.type === 'text'
              ? this.parseText()
              : this.tok();
          }
    
          return this.renderer.listitem(body);
        }
        case 'loose_item_start': {
          body = '';
    
          while (this.next().type !== 'list_item_end') {
            body += this.tok();
          }
    
          return this.renderer.listitem(body);
        }
        case 'html': {
          // TODO parse inline content if parameter markdown=1
          return this.renderer.html(this.token.text);
        }
        case 'paragraph': {
          return this.renderer.paragraph(this.inline.output(this.token.text));
        }
        case 'text': {
          return this.renderer.paragraph(this.parseText());
        }
      }
    };
    
    /**
     * Helpers
     */
    
    function escape(html, encode) {
      return html
        .replace(!encode ? /&(?!#?\w+;)/g : /&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }
    
    function unescape(html) {
      // explicitly match decimal, hex, and named HTML entities
      return html.replace(/&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig, function(_, n) {
        n = n.toLowerCase();
        if (n === 'colon') return ':';
        if (n.charAt(0) === '#') {
          return n.charAt(1) === 'x'
            ? String.fromCharCode(parseInt(n.substring(2), 16))
            : String.fromCharCode(+n.substring(1));
        }
        return '';
      });
    }
    
    function edit(regex, opt) {
      regex = regex.source || regex;
      opt = opt || '';
      return {
        replace: function(name, val) {
          val = val.source || val;
          val = val.replace(/(^|[^\[])\^/g, '$1');
          regex = regex.replace(name, val);
          return this;
        },
        getRegex: function() {
          return new RegExp(regex, opt);
        }
      };
    }
    
    function resolveUrl(base, href) {
      if (!baseUrls[' ' + base]) {
        // we can ignore everything in base after the last slash of its path component,
        // but we might need to add _that_
        // https://tools.ietf.org/html/rfc3986#section-3
        if (/^[^:]+:\/*[^/]*$/.test(base)) {
          baseUrls[' ' + base] = base + '/';
        } else {
          baseUrls[' ' + base] = base.replace(/[^/]*$/, '');
        }
      }
      base = baseUrls[' ' + base];
    
      if (href.slice(0, 2) === '//') {
        return base.replace(/:[\s\S]*/, ':') + href;
      } else if (href.charAt(0) === '/') {
        return base.replace(/(:\/*[^/]*)[\s\S]*/, '$1') + href;
      } else {
        return base + href;
      }
    }
    var baseUrls = {};
    var originIndependentUrl = /^$|^[a-z][a-z0-9+.-]*:|^[?#]/i;
    
    function noop() {}
    noop.exec = noop;
    
    function merge(obj) {
      var i = 1,
          target,
          key;
    
      for (; i < arguments.length; i++) {
        target = arguments[i];
        for (key in target) {
          if (Object.prototype.hasOwnProperty.call(target, key)) {
            obj[key] = target[key];
          }
        }
      }
    
      return obj;
    }
    
    function splitCells(tableRow, count) {
      var cells = tableRow.replace(/([^\\])\|/g, '$1 |').split(/ +\| */),
          i = 0;
    
      if (cells.length > count) {
        cells.splice(count);
      } else {
        while (cells.length < count) cells.push('');
      }
    
      for (; i < cells.length; i++) {
        cells[i] = cells[i].replace(/\\\|/g, '|');
      }
      return cells;
    }
    
    /**
     * Marked
     */
    
    function marked(src, opt, callback) {
      // throw error in case of non string input
      if (typeof src === 'undefined' || src === null) {
        throw new Error('marked(): input parameter is undefined or null');
      }
      if (typeof src !== 'string') {
        throw new Error('marked(): input parameter is of type '
          + Object.prototype.toString.call(src) + ', string expected');
      }
    
      if (callback || typeof opt === 'function') {
        if (!callback) {
          callback = opt;
          opt = null;
        }
    
        opt = merge({}, marked.defaults, opt || {});
    
        var highlight = opt.highlight,
            tokens,
            pending,
            i = 0;
    
        try {
          tokens = Lexer.lex(src, opt)
        } catch (e) {
          return callback(e);
        }
    
        pending = tokens.length;
    
        var done = function(err) {
          if (err) {
            opt.highlight = highlight;
            return callback(err);
          }
    
          var out;
    
          try {
            out = Parser.parse(tokens, opt);
          } catch (e) {
            err = e;
          }
    
          opt.highlight = highlight;
    
          return err
            ? callback(err)
            : callback(null, out);
        };
    
        if (!highlight || highlight.length < 3) {
          return done();
        }
    
        delete opt.highlight;
    
        if (!pending) return done();
    
        for (; i < tokens.length; i++) {
          (function(token) {
            if (token.type !== 'code') {
              return --pending || done();
            }
            return highlight(token.text, token.lang, function(err, code) {
              if (err) return done(err);
              if (code == null || code === token.text) {
                return --pending || done();
              }
              token.text = code;
              token.escaped = true;
              --pending || done();
            });
          })(tokens[i]);
        }
    
        return;
      }
      try {
        if (opt) opt = merge({}, marked.defaults, opt);
        return Parser.parse(Lexer.lex(src, opt), opt);
      } catch (e) {
        e.message += '\nPlease report this to https://github.com/markedjs/marked.';
        if ((opt || marked.defaults).silent) {
          return '<p>An error occurred:</p><pre>'
            + escape(e.message + '', true)
            + '</pre>';
        }
        throw e;
      }
    }
    
    /**
     * Options
     */
    
    marked.options =
    marked.setOptions = function(opt) {
      merge(marked.defaults, opt);
      return marked;
    };
    
    marked.getDefaults = function () {
      return {
        baseUrl: null,
        breaks: false,
        gfm: true,
        headerIds: true,
        headerPrefix: '',
        highlight: null,
        langPrefix: 'language-',
        mangle: true,
        pedantic: false,
        renderer: new Renderer(),
        sanitize: false,
        sanitizer: null,
        silent: false,
        smartLists: false,
        smartypants: false,
        tables: true,
        xhtml: false
      };
    }
    
    marked.defaults = marked.getDefaults();
    
    /**
     * Expose
     */
    
    marked.Parser = Parser;
    marked.parser = Parser.parse;
    
    marked.Renderer = Renderer;
    marked.TextRenderer = TextRenderer;
    
    marked.Lexer = Lexer;
    marked.lexer = Lexer.lex;
    
    marked.InlineLexer = InlineLexer;
    marked.inlineLexer = InlineLexer.output;
    
    marked.parse = marked;
    
    {
      module.exports = marked;
    }
    })(commonjsGlobal || (typeof window !== 'undefined' ? window : commonjsGlobal));
    });
    
    var prism = createCommonjsModule(function (module) {
    /* **********************************************
         Begin prism-core.js
    ********************************************** */
    
    var _self = (typeof window !== 'undefined')
        ? window   // if in browser
        : (
            (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope)
            ? self // if in worker
            : {}   // if in node js
        );
    
    /**
     * Prism: Lightweight, robust, elegant syntax highlighting
     * MIT license http://www.opensource.org/licenses/mit-license.php/
     * @author Lea Verou http://lea.verou.me
     */
    
    var Prism = (function(){
    
    // Private helper vars
    var lang = /\blang(?:uage)?-([\w-]+)\b/i;
    var uniqueId = 0;
    
    var _ = _self.Prism = {
        manual: _self.Prism && _self.Prism.manual,
        disableWorkerMessageHandler: _self.Prism && _self.Prism.disableWorkerMessageHandler,
        util: {
            encode: function (tokens) {
                if (tokens instanceof Token) {
                    return new Token(tokens.type, _.util.encode(tokens.content), tokens.alias);
                } else if (_.util.type(tokens) === 'Array') {
                    return tokens.map(_.util.encode);
                } else {
                    return tokens.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\u00a0/g, ' ');
                }
            },
    
            type: function (o) {
                return Object.prototype.toString.call(o).match(/\[object (\w+)\]/)[1];
            },
    
            objId: function (obj) {
                if (!obj['__id']) {
                    Object.defineProperty(obj, '__id', { value: ++uniqueId });
                }
                return obj['__id'];
            },
    
            // Deep clone a language definition (e.g. to extend it)
            clone: function (o, visited) {
                var type = _.util.type(o);
                visited = visited || {};
    
                switch (type) {
                    case 'Object':
                        if (visited[_.util.objId(o)]) {
                            return visited[_.util.objId(o)];
                        }
                        var clone = {};
                        visited[_.util.objId(o)] = clone;
    
                        for (var key in o) {
                            if (o.hasOwnProperty(key)) {
                                clone[key] = _.util.clone(o[key], visited);
                            }
                        }
    
                        return clone;
    
                    case 'Array':
                        if (visited[_.util.objId(o)]) {
                            return visited[_.util.objId(o)];
                        }
                        var clone = [];
                        visited[_.util.objId(o)] = clone;
    
                        o.forEach(function (v, i) {
                            clone[i] = _.util.clone(v, visited);
                        });
    
                        return clone;
                }
    
                return o;
            }
        },
    
        languages: {
            extend: function (id, redef) {
                var lang = _.util.clone(_.languages[id]);
    
                for (var key in redef) {
                    lang[key] = redef[key];
                }
    
                return lang;
            },
    
            /**
             * Insert a token before another token in a language literal
             * As this needs to recreate the object (we cannot actually insert before keys in object literals),
             * we cannot just provide an object, we need anobject and a key.
             * @param inside The key (or language id) of the parent
             * @param before The key to insert before. If not provided, the function appends instead.
             * @param insert Object with the key/value pairs to insert
             * @param root The object that contains `inside`. If equal to Prism.languages, it can be omitted.
             */
            insertBefore: function (inside, before, insert, root) {
                root = root || _.languages;
                var grammar = root[inside];
    
                if (arguments.length == 2) {
                    insert = arguments[1];
    
                    for (var newToken in insert) {
                        if (insert.hasOwnProperty(newToken)) {
                            grammar[newToken] = insert[newToken];
                        }
                    }
    
                    return grammar;
                }
    
                var ret = {};
    
                for (var token in grammar) {
    
                    if (grammar.hasOwnProperty(token)) {
    
                        if (token == before) {
    
                            for (var newToken in insert) {
    
                                if (insert.hasOwnProperty(newToken)) {
                                    ret[newToken] = insert[newToken];
                                }
                            }
                        }
    
                        ret[token] = grammar[token];
                    }
                }
    
                // Update references in other language definitions
                _.languages.DFS(_.languages, function(key, value) {
                    if (value === root[inside] && key != inside) {
                        this[key] = ret;
                    }
                });
    
                return root[inside] = ret;
            },
    
            // Traverse a language definition with Depth First Search
            DFS: function(o, callback, type, visited) {
                visited = visited || {};
                for (var i in o) {
                    if (o.hasOwnProperty(i)) {
                        callback.call(o, i, o[i], type || i);
    
                        if (_.util.type(o[i]) === 'Object' && !visited[_.util.objId(o[i])]) {
                            visited[_.util.objId(o[i])] = true;
                            _.languages.DFS(o[i], callback, null, visited);
                        }
                        else if (_.util.type(o[i]) === 'Array' && !visited[_.util.objId(o[i])]) {
                            visited[_.util.objId(o[i])] = true;
                            _.languages.DFS(o[i], callback, i, visited);
                        }
                    }
                }
            }
        },
        plugins: {},
    
        highlightAll: function(async, callback) {
            _.highlightAllUnder(document, async, callback);
        },
    
        highlightAllUnder: function(container, async, callback) {
            var env = {
                callback: callback,
                selector: 'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code'
            };
    
            _.hooks.run("before-highlightall", env);
    
            var elements = env.elements || container.querySelectorAll(env.selector);
    
            for (var i=0, element; element = elements[i++];) {
                _.highlightElement(element, async === true, env.callback);
            }
        },
    
        highlightElement: function(element, async, callback) {
            // Find language
            var language, grammar, parent = element;
    
            while (parent && !lang.test(parent.className)) {
                parent = parent.parentNode;
            }
    
            if (parent) {
                language = (parent.className.match(lang) || [,''])[1].toLowerCase();
                grammar = _.languages[language];
            }
    
            // Set language on the element, if not present
            element.className = element.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;
    
            if (element.parentNode) {
                // Set language on the parent, for styling
                parent = element.parentNode;
    
                if (/pre/i.test(parent.nodeName)) {
                    parent.className = parent.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;
                }
            }
    
            var code = element.textContent;
    
            var env = {
                element: element,
                language: language,
                grammar: grammar,
                code: code
            };
    
            _.hooks.run('before-sanity-check', env);
    
            if (!env.code || !env.grammar) {
                if (env.code) {
                    _.hooks.run('before-highlight', env);
                    env.element.textContent = env.code;
                    _.hooks.run('after-highlight', env);
                }
                _.hooks.run('complete', env);
                return;
            }
    
            _.hooks.run('before-highlight', env);
    
            if (async && _self.Worker) {
                var worker = new Worker(_.filename);
    
                worker.onmessage = function(evt) {
                    env.highlightedCode = evt.data;
    
                    _.hooks.run('before-insert', env);
    
                    env.element.innerHTML = env.highlightedCode;
    
                    callback && callback.call(env.element);
                    _.hooks.run('after-highlight', env);
                    _.hooks.run('complete', env);
                };
    
                worker.postMessage(JSON.stringify({
                    language: env.language,
                    code: env.code,
                    immediateClose: true
                }));
            }
            else {
                env.highlightedCode = _.highlight(env.code, env.grammar, env.language);
    
                _.hooks.run('before-insert', env);
    
                env.element.innerHTML = env.highlightedCode;
    
                callback && callback.call(element);
    
                _.hooks.run('after-highlight', env);
                _.hooks.run('complete', env);
            }
        },
    
        highlight: function (text, grammar, language) {
            var env = {
                code: text,
                grammar: grammar,
                language: language
            };
            _.hooks.run('before-tokenize', env);
            env.tokens = _.tokenize(env.code, env.grammar);
            _.hooks.run('after-tokenize', env);
            return Token.stringify(_.util.encode(env.tokens), env.language);
        },
    
        matchGrammar: function (text, strarr, grammar, index, startPos, oneshot, target) {
            var Token = _.Token;
    
            for (var token in grammar) {
                if(!grammar.hasOwnProperty(token) || !grammar[token]) {
                    continue;
                }
    
                if (token == target) {
                    return;
                }
    
                var patterns = grammar[token];
                patterns = (_.util.type(patterns) === "Array") ? patterns : [patterns];
    
                for (var j = 0; j < patterns.length; ++j) {
                    var pattern = patterns[j],
                        inside = pattern.inside,
                        lookbehind = !!pattern.lookbehind,
                        greedy = !!pattern.greedy,
                        lookbehindLength = 0,
                        alias = pattern.alias;
    
                    if (greedy && !pattern.pattern.global) {
                        // Without the global flag, lastIndex won't work
                        var flags = pattern.pattern.toString().match(/[imuy]*$/)[0];
                        pattern.pattern = RegExp(pattern.pattern.source, flags + "g");
                    }
    
                    pattern = pattern.pattern || pattern;
    
                    // Don’t cache length as it changes during the loop
                    for (var i = index, pos = startPos; i < strarr.length; pos += strarr[i].length, ++i) {
    
                        var str = strarr[i];
    
                        if (strarr.length > text.length) {
                            // Something went terribly wrong, ABORT, ABORT!
                            return;
                        }
    
                        if (str instanceof Token) {
                            continue;
                        }
    
                        if (greedy && i != strarr.length - 1) {
                            pattern.lastIndex = pos;
                            var match = pattern.exec(text);
                            if (!match) {
                                break;
                            }
    
                            var from = match.index + (lookbehind ? match[1].length : 0),
                                to = match.index + match[0].length,
                                k = i,
                                p = pos;
    
                            for (var len = strarr.length; k < len && (p < to || (!strarr[k].type && !strarr[k - 1].greedy)); ++k) {
                                p += strarr[k].length;
                                // Move the index i to the element in strarr that is closest to from
                                if (from >= p) {
                                    ++i;
                                    pos = p;
                                }
                            }
    
                            // If strarr[i] is a Token, then the match starts inside another Token, which is invalid
                            if (strarr[i] instanceof Token) {
                                continue;
                            }
    
                            // Number of tokens to delete and replace with the new match
                            delNum = k - i;
                            str = text.slice(pos, p);
                            match.index -= pos;
                        } else {
                            pattern.lastIndex = 0;
    
                            var match = pattern.exec(str),
                                delNum = 1;
                        }
    
                        if (!match) {
                            if (oneshot) {
                                break;
                            }
    
                            continue;
                        }
    
                        if(lookbehind) {
                            lookbehindLength = match[1] ? match[1].length : 0;
                        }
    
                        var from = match.index + lookbehindLength,
                            match = match[0].slice(lookbehindLength),
                            to = from + match.length,
                            before = str.slice(0, from),
                            after = str.slice(to);
    
                        var args = [i, delNum];
    
                        if (before) {
                            ++i;
                            pos += before.length;
                            args.push(before);
                        }
    
                        var wrapped = new Token(token, inside? _.tokenize(match, inside) : match, alias, match, greedy);
    
                        args.push(wrapped);
    
                        if (after) {
                            args.push(after);
                        }
    
                        Array.prototype.splice.apply(strarr, args);
    
                        if (delNum != 1)
                            { _.matchGrammar(text, strarr, grammar, i, pos, true, token); }
    
                        if (oneshot)
                            { break; }
                    }
                }
            }
        },
    
        tokenize: function(text, grammar, language) {
            var strarr = [text];
    
            var rest = grammar.rest;
    
            if (rest) {
                for (var token in rest) {
                    grammar[token] = rest[token];
                }
    
                delete grammar.rest;
            }
    
            _.matchGrammar(text, strarr, grammar, 0, 0, false);
    
            return strarr;
        },
    
        hooks: {
            all: {},
    
            add: function (name, callback) {
                var hooks = _.hooks.all;
    
                hooks[name] = hooks[name] || [];
    
                hooks[name].push(callback);
            },
    
            run: function (name, env) {
                var callbacks = _.hooks.all[name];
    
                if (!callbacks || !callbacks.length) {
                    return;
                }
    
                for (var i=0, callback; callback = callbacks[i++];) {
                    callback(env);
                }
            }
        }
    };
    
    var Token = _.Token = function(type, content, alias, matchedStr, greedy) {
        this.type = type;
        this.content = content;
        this.alias = alias;
        // Copy of the full string this token was created from
        this.length = (matchedStr || "").length|0;
        this.greedy = !!greedy;
    };
    
    Token.stringify = function(o, language, parent) {
        if (typeof o == 'string') {
            return o;
        }
    
        if (_.util.type(o) === 'Array') {
            return o.map(function(element) {
                return Token.stringify(element, language, o);
            }).join('');
        }
    
        var env = {
            type: o.type,
            content: Token.stringify(o.content, language, parent),
            tag: 'span',
            classes: ['token', o.type],
            attributes: {},
            language: language,
            parent: parent
        };
    
        if (o.alias) {
            var aliases = _.util.type(o.alias) === 'Array' ? o.alias : [o.alias];
            Array.prototype.push.apply(env.classes, aliases);
        }
    
        _.hooks.run('wrap', env);
    
        var attributes = Object.keys(env.attributes).map(function(name) {
            return name + '="' + (env.attributes[name] || '').replace(/"/g, '&quot;') + '"';
        }).join(' ');
    
        return '<' + env.tag + ' class="' + env.classes.join(' ') + '"' + (attributes ? ' ' + attributes : '') + '>' + env.content + '</' + env.tag + '>';
    
    };
    
    if (!_self.document) {
        if (!_self.addEventListener) {
            // in Node.js
            return _self.Prism;
        }
    
        if (!_.disableWorkerMessageHandler) {
            // In worker
            _self.addEventListener('message', function (evt) {
                var message = JSON.parse(evt.data),
                    lang = message.language,
                    code = message.code,
                    immediateClose = message.immediateClose;
    
                _self.postMessage(_.highlight(code, _.languages[lang], lang));
                if (immediateClose) {
                    _self.close();
                }
            }, false);
        }
    
        return _self.Prism;
    }
    
    //Get current script and highlight
    var script = document.currentScript || [].slice.call(document.getElementsByTagName("script")).pop();
    
    if (script) {
        _.filename = script.src;
    
        if (!_.manual && !script.hasAttribute('data-manual')) {
            if(document.readyState !== "loading") {
                if (window.requestAnimationFrame) {
                    window.requestAnimationFrame(_.highlightAll);
                } else {
                    window.setTimeout(_.highlightAll, 16);
                }
            }
            else {
                document.addEventListener('DOMContentLoaded', _.highlightAll);
            }
        }
    }
    
    return _self.Prism;
    
    })();
    
    if ('object' !== 'undefined' && module.exports) {
        module.exports = Prism;
    }
    
    // hack for components to work correctly in node.js
    if (typeof commonjsGlobal !== 'undefined') {
        commonjsGlobal.Prism = Prism;
    }
    
    
    /* **********************************************
         Begin prism-markup.js
    ********************************************** */
    
    Prism.languages.markup = {
        'comment': /<!--[\s\S]*?-->/,
        'prolog': /<\?[\s\S]+?\?>/,
        'doctype': /<!DOCTYPE[\s\S]+?>/i,
        'cdata': /<!\[CDATA\[[\s\S]*?]]>/i,
        'tag': {
            pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s+[^\s>\/=]+(?:=(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s'">=]+))?)*\s*\/?>/i,
            greedy: true,
            inside: {
                'tag': {
                    pattern: /^<\/?[^\s>\/]+/i,
                    inside: {
                        'punctuation': /^<\/?/,
                        'namespace': /^[^\s>\/:]+:/
                    }
                },
                'attr-value': {
                    pattern: /=(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s'">=]+)/i,
                    inside: {
                        'punctuation': [
                            /^=/,
                            {
                                pattern: /(^|[^\\])["']/,
                                lookbehind: true
                            }
                        ]
                    }
                },
                'punctuation': /\/?>/,
                'attr-name': {
                    pattern: /[^\s>\/]+/,
                    inside: {
                        'namespace': /^[^\s>\/:]+:/
                    }
                }
    
            }
        },
        'entity': /&#?[\da-z]{1,8};/i
    };
    
    Prism.languages.markup['tag'].inside['attr-value'].inside['entity'] =
        Prism.languages.markup['entity'];
    
    // Plugin to make entity title show the real entity, idea by Roman Komarov
    Prism.hooks.add('wrap', function(env) {
    
        if (env.type === 'entity') {
            env.attributes['title'] = env.content.replace(/&amp;/, '&');
        }
    });
    
    Prism.languages.xml = Prism.languages.markup;
    Prism.languages.html = Prism.languages.markup;
    Prism.languages.mathml = Prism.languages.markup;
    Prism.languages.svg = Prism.languages.markup;
    
    
    /* **********************************************
         Begin prism-css.js
    ********************************************** */
    
    Prism.languages.css = {
        'comment': /\/\*[\s\S]*?\*\//,
        'atrule': {
            pattern: /@[\w-]+?.*?(?:;|(?=\s*\{))/i,
            inside: {
                'rule': /@[\w-]+/
                // See rest below
            }
        },
        'url': /url\((?:(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1|.*?)\)/i,
        'selector': /[^{}\s][^{};]*?(?=\s*\{)/,
        'string': {
            pattern: /("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
            greedy: true
        },
        'property': /[-_a-z\xA0-\uFFFF][-\w\xA0-\uFFFF]*(?=\s*:)/i,
        'important': /\B!important\b/i,
        'function': /[-a-z0-9]+(?=\()/i,
        'punctuation': /[(){};:]/
    };
    
    Prism.languages.css['atrule'].inside.rest = Prism.languages.css;
    
    if (Prism.languages.markup) {
        Prism.languages.insertBefore('markup', 'tag', {
            'style': {
                pattern: /(<style[\s\S]*?>)[\s\S]*?(?=<\/style>)/i,
                lookbehind: true,
                inside: Prism.languages.css,
                alias: 'language-css',
                greedy: true
            }
        });
    
        Prism.languages.insertBefore('inside', 'attr-value', {
            'style-attr': {
                pattern: /\s*style=("|')(?:\\[\s\S]|(?!\1)[^\\])*\1/i,
                inside: {
                    'attr-name': {
                        pattern: /^\s*style/i,
                        inside: Prism.languages.markup.tag.inside
                    },
                    'punctuation': /^\s*=\s*['"]|['"]\s*$/,
                    'attr-value': {
                        pattern: /.+/i,
                        inside: Prism.languages.css
                    }
                },
                alias: 'language-css'
            }
        }, Prism.languages.markup.tag);
    }
    
    /* **********************************************
         Begin prism-clike.js
    ********************************************** */
    
    Prism.languages.clike = {
        'comment': [
            {
                pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
                lookbehind: true
            },
            {
                pattern: /(^|[^\\:])\/\/.*/,
                lookbehind: true,
                greedy: true
            }
        ],
        'string': {
            pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
            greedy: true
        },
        'class-name': {
            pattern: /((?:\b(?:class|interface|extends|implements|trait|instanceof|new)\s+)|(?:catch\s+\())[\w.\\]+/i,
            lookbehind: true,
            inside: {
                punctuation: /[.\\]/
            }
        },
        'keyword': /\b(?:if|else|while|do|for|return|in|instanceof|function|new|try|throw|catch|finally|null|break|continue)\b/,
        'boolean': /\b(?:true|false)\b/,
        'function': /[a-z0-9_]+(?=\()/i,
        'number': /\b0x[\da-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:e[+-]?\d+)?/i,
        'operator': /--?|\+\+?|!=?=?|<=?|>=?|==?=?|&&?|\|\|?|\?|\*|\/|~|\^|%/,
        'punctuation': /[{}[\];(),.:]/
    };
    
    
    /* **********************************************
         Begin prism-javascript.js
    ********************************************** */
    
    Prism.languages.javascript = Prism.languages.extend('clike', {
        'keyword': /\b(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|var|void|while|with|yield)\b/,
        'number': /\b(?:0[xX][\dA-Fa-f]+|0[bB][01]+|0[oO][0-7]+|NaN|Infinity)\b|(?:\b\d+\.?\d*|\B\.\d+)(?:[Ee][+-]?\d+)?/,
        // Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
        'function': /[_$a-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*\()/i,
        'operator': /-[-=]?|\+[+=]?|!=?=?|<<?=?|>>?>?=?|=(?:==?|>)?|&[&=]?|\|[|=]?|\*\*?=?|\/=?|~|\^=?|%=?|\?|\.{3}/
    });
    
    Prism.languages.insertBefore('javascript', 'keyword', {
        'regex': {
            pattern: /((?:^|[^$\w\xA0-\uFFFF."'\])\s])\s*)\/(\[[^\]\r\n]+]|\\.|[^/\\\[\r\n])+\/[gimyu]{0,5}(?=\s*($|[\r\n,.;})\]]))/,
            lookbehind: true,
            greedy: true
        },
        // This must be declared before keyword because we use "function" inside the look-forward
        'function-variable': {
            pattern: /[_$a-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*=\s*(?:function\b|(?:\([^()]*\)|[_$a-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*)\s*=>))/i,
            alias: 'function'
        },
        'constant': /\b[A-Z][A-Z\d_]*\b/
    });
    
    Prism.languages.insertBefore('javascript', 'string', {
        'template-string': {
            pattern: /`(?:\\[\s\S]|\${[^}]+}|[^\\`])*`/,
            greedy: true,
            inside: {
                'interpolation': {
                    pattern: /\${[^}]+}/,
                    inside: {
                        'interpolation-punctuation': {
                            pattern: /^\${|}$/,
                            alias: 'punctuation'
                        },
                        rest: null // See below
                    }
                },
                'string': /[\s\S]+/
            }
        }
    });
    Prism.languages.javascript['template-string'].inside['interpolation'].inside.rest = Prism.languages.javascript;
    
    if (Prism.languages.markup) {
        Prism.languages.insertBefore('markup', 'tag', {
            'script': {
                pattern: /(<script[\s\S]*?>)[\s\S]*?(?=<\/script>)/i,
                lookbehind: true,
                inside: Prism.languages.javascript,
                alias: 'language-javascript',
                greedy: true
            }
        });
    }
    
    Prism.languages.js = Prism.languages.javascript;
    
    
    /* **********************************************
         Begin prism-file-highlight.js
    ********************************************** */
    
    (function () {
        if (typeof self === 'undefined' || !self.Prism || !self.document || !document.querySelector) {
            return;
        }
    
        self.Prism.fileHighlight = function() {
    
            var Extensions = {
                'js': 'javascript',
                'py': 'python',
                'rb': 'ruby',
                'ps1': 'powershell',
                'psm1': 'powershell',
                'sh': 'bash',
                'bat': 'batch',
                'h': 'c',
                'tex': 'latex'
            };
    
            Array.prototype.slice.call(document.querySelectorAll('pre[data-src]')).forEach(function (pre) {
                var src = pre.getAttribute('data-src');
    
                var language, parent = pre;
                var lang = /\blang(?:uage)?-([\w-]+)\b/i;
                while (parent && !lang.test(parent.className)) {
                    parent = parent.parentNode;
                }
    
                if (parent) {
                    language = (pre.className.match(lang) || [, ''])[1];
                }
    
                if (!language) {
                    var extension = (src.match(/\.(\w+)$/) || [, ''])[1];
                    language = Extensions[extension] || extension;
                }
    
                var code = document.createElement('code');
                code.className = 'language-' + language;
    
                pre.textContent = '';
    
                code.textContent = 'Loading…';
    
                pre.appendChild(code);
    
                var xhr = new XMLHttpRequest();
    
                xhr.open('GET', src, true);
    
                xhr.onreadystatechange = function () {
                    if (xhr.readyState == 4) {
    
                        if (xhr.status < 400 && xhr.responseText) {
                            code.textContent = xhr.responseText;
    
                            Prism.highlightElement(code);
                        }
                        else if (xhr.status >= 400) {
                            code.textContent = '✖ Error ' + xhr.status + ' while fetching file: ' + xhr.statusText;
                        }
                        else {
                            code.textContent = '✖ Error: File does not exist or is empty';
                        }
                    }
                };
    
                xhr.send(null);
            });
    
            if (Prism.plugins.toolbar) {
                Prism.plugins.toolbar.registerButton('download-file', function (env) {
                    var pre = env.element.parentNode;
                    if (!pre || !/pre/i.test(pre.nodeName) || !pre.hasAttribute('data-src') || !pre.hasAttribute('data-download-link')) {
                        return;
                    }
                    var src = pre.getAttribute('data-src');
                    var a = document.createElement('a');
                    a.textContent = pre.getAttribute('data-download-link-label') || 'Download';
                    a.setAttribute('download', '');
                    a.href = src;
                    return a;
                });
            }
    
        };
    
        document.addEventListener('DOMContentLoaded', self.Prism.fileHighlight);
    
    })();
    });
    
    /**
     * Gen toc tree
     * @link https://github.com/killercup/grock/blob/5280ae63e16c5739e9233d9009bc235ed7d79a50/styles/solarized/assets/js/behavior.coffee#L54-L81
     * @param  {Array} toc
     * @param  {Number} maxLevel
     * @return {Array}
     */
    function genTree(toc, maxLevel) {
      var headlines = [];
      var last = {};
    
      toc.forEach(function (headline) {
        var level = headline.level || 1;
        var len = level - 1;
    
        if (level > maxLevel) {
          return
        }
        if (last[len]) {
          last[len].children = (last[len].children || []).concat(headline);
        } else {
          headlines.push(headline);
        }
        last[level] = headline;
      });
    
      return headlines
    }
    
    var cache$1 = {};
    var re = /[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,./:;<=>?@[\]^`{|}~]/g;
    
    function lower(string) {
      return string.toLowerCase()
    }
    
    function slugify(str) {
      if (typeof str !== 'string') {
        return ''
      }
    
      var slug = str
        .trim()
        .replace(/[A-Z]+/g, lower)
        .replace(/<[^>\d]+>/g, '')
        .replace(re, '')
        .replace(/\s/g, '-')
        .replace(/-+/g, '-')
        .replace(/^(\d)/, '_$1');
      var count = cache$1[slug];
    
      count = hasOwn.call(cache$1, slug) ? count + 1 : 0;
      cache$1[slug] = count;
    
      if (count) {
        slug = slug + '-' + count;
      }
    
      return slug
    }
    
    slugify.clear = function () {
      cache$1 = {};
    };
    
    function replace(m, $1) {
      return '<img class="emoji" src="https://github.githubassets.com/images/icons/emoji/' + $1 + '.png" alt="' + $1 + '" />'
    }
    
    function emojify(text) {
      return text
        .replace(/<(pre|template|code)[^>]*?>[\s\S]+?<\/(pre|template|code)>/g, function (m) { return m.replace(/:/g, '__colon__'); })
        .replace(/:(\w+?):/ig, (inBrowser && window.emojify) || replace)
        .replace(/__colon__/g, ':')
    }
    
    var decode = decodeURIComponent;
    var encode = encodeURIComponent;
    
    function parseQuery(query) {
      var res = {};
    
      query = query.trim().replace(/^(\?|#|&)/, '');
    
      if (!query) {
        return res
      }
    
      // Simple parse
      query.split('&').forEach(function (param) {
        var parts = param.replace(/\+/g, ' ').split('=');
    
        res[parts[0]] = parts[1] && decode(parts[1]);
      });
    
      return res
    }
    
    function stringifyQuery(obj, ignores) {
      if ( ignores === void 0 ) ignores = [];
    
      var qs = [];
    
      for (var key in obj) {
        if (ignores.indexOf(key) > -1) {
          continue
        }
        qs.push(
          obj[key] ?
            ((encode(key)) + "=" + (encode(obj[key]))).toLowerCase() :
            encode(key)
        );
      }
    
      return qs.length ? ("?" + (qs.join('&'))) : ''
    }
    
    var isAbsolutePath = cached(function (path) {
      return /(:|(\/{2}))/g.test(path)
    });
    
    var getParentPath = cached(function (path) {
      return /\/$/g.test(path) ?
        path :
        (path = path.match(/(\S*\/)[^/]+$/)) ? path[1] : ''
    });
    
    var cleanPath = cached(function (path) {
      return path.replace(/^\/+/, '/').replace(/([^:])\/{2,}/g, '$1/')
    });
    
    var resolvePath = cached(function (path) {
      var segments = path.replace(/^\//, '').split('/');
      var resolved = [];
      for (var i = 0, len = segments.length; i < len; i++) {
        var segment = segments[i];
        if (segment === '..') {
          resolved.pop();
        } else if (segment !== '.') {
          resolved.push(segment);
        }
      }
      return '/' + resolved.join('/')
    });
    
    function getPath() {
      var args = [], len = arguments.length;
      while ( len-- ) args[ len ] = arguments[ len ];
    
      return cleanPath(args.join('/'))
    }
    
    var replaceSlug = cached(function (path) {
      return path.replace('#', '?id=')
    });
    
    Prism.languages['markup-templating'] = {};
    
    Object.defineProperties(Prism.languages['markup-templating'], {
        buildPlaceholders: {
            // Tokenize all inline templating expressions matching placeholderPattern
            // If the replaceFilter function is provided, it will be called with every match.
            // If it returns false, the match will not be replaced.
            value: function (env, language, placeholderPattern, replaceFilter) {
                if (env.language !== language) {
                    return;
                }
    
                env.tokenStack = [];
    
                env.code = env.code.replace(placeholderPattern, function(match) {
                    if (typeof replaceFilter === 'function' && !replaceFilter(match)) {
                        return match;
                    }
                    var i = env.tokenStack.length;
                    // Check for existing strings
                    while (env.code.indexOf('___' + language.toUpperCase() + i + '___') !== -1)
                        { ++i; }
    
                    // Create a sparse array
                    env.tokenStack[i] = match;
    
                    return '___' + language.toUpperCase() + i + '___';
                });
    
                // Switch the grammar to markup
                env.grammar = Prism.languages.markup;
            }
        },
        tokenizePlaceholders: {
            // Replace placeholders with proper tokens after tokenizing
            value: function (env, language) {
                if (env.language !== language || !env.tokenStack) {
                    return;
                }
    
                // Switch the grammar back
                env.grammar = Prism.languages[language];
    
                var j = 0;
                var keys = Object.keys(env.tokenStack);
                var walkTokens = function (tokens) {
                    if (j >= keys.length) {
                        return;
                    }
                    for (var i = 0; i < tokens.length; i++) {
                        var token = tokens[i];
                        if (typeof token === 'string' || (token.content && typeof token.content === 'string')) {
                            var k = keys[j];
                            var t = env.tokenStack[k];
                            var s = typeof token === 'string' ? token : token.content;
    
                            var index = s.indexOf('___' + language.toUpperCase() + k + '___');
                            if (index > -1) {
                                ++j;
                                var before = s.substring(0, index);
                                var middle = new Prism.Token(language, Prism.tokenize(t, env.grammar, language), 'language-' + language, t);
                                var after = s.substring(index + ('___' + language.toUpperCase() + k + '___').length);
                                var replacement;
                                if (before || after) {
                                    replacement = [before, middle, after].filter(function (v) { return !!v; });
                                    walkTokens(replacement);
                                } else {
                                    replacement = middle;
                                }
                                if (typeof token === 'string') {
                                    Array.prototype.splice.apply(tokens, [i, 1].concat(replacement));
                                } else {
                                    token.content = replacement;
                                }
    
                                if (j >= keys.length) {
                                    break;
                                }
                            }
                        } else if (token.content && typeof token.content !== 'string') {
                            walkTokens(token.content);
                        }
                    }
                };
    
                walkTokens(env.tokens);
            }
        }
    });
    
    // See https://github.com/PrismJS/prism/pull/1367
    var cachedLinks = {};
    
    function getAndRemoveConfig(str) {
      if ( str === void 0 ) str = '';
    
      var config = {};
    
      if (str) {
        str = str
          .replace(/^'/, '')
          .replace(/'$/, '')
          .replace(/(?:^|\s):([\w-]+)=?([\w-]+)?/g, function (m, key, value) {
            config[key] = (value && value.replace(/&quot;/g, '')) || true;
            return ''
          })
          .trim();
      }
    
      return {str: str, config: config}
    }
    
    var compileMedia = {
      markdown: function markdown(url) {
        return {
          url: url
        }
      },
      mermaid: function mermaid(url) {
        return {
          url: url
        }
      },
      iframe: function iframe(url, title) {
        return {
          html: ("<iframe src=\"" + url + "\" " + (title || 'width=100% height=400') + "></iframe>")
        }
      },
      video: function video(url, title) {
        return {
          html: ("<video src=\"" + url + "\" " + (title || 'controls') + ">Not Support</video>")
        }
      },
      audio: function audio(url, title) {
        return {
          html: ("<audio src=\"" + url + "\" " + (title || 'controls') + ">Not Support</audio>")
        }
      },
      code: function code(url, title) {
        var lang = url.match(/\.(\w+)$/);
    
        lang = title || (lang && lang[1]);
        if (lang === 'md') {
          lang = 'markdown';
        }
    
        return {
          url: url,
          lang: lang
        }
      }
    };
    
    var Compiler = function Compiler(config, router) {
      var this$1 = this;
    
      this.config = config;
      this.router = router;
      this.cacheTree = {};
      this.toc = [];
      this.cacheTOC = {};
      this.linkTarget = config.externalLinkTarget || '_blank';
      this.contentBase = router.getBasePath();
    
      var renderer = this._initRenderer();
      var compile;
      var mdConf = config.markdown || {};
    
      if (isFn(mdConf)) {
        compile = mdConf(marked, renderer);
      } else {
        marked.setOptions(
          merge(mdConf, {
            renderer: merge(renderer, mdConf.renderer)
          })
        );
        compile = marked;
      }
    
      this._marked = compile;
      this.compile = function (text) {
        var isCached = true;
        var result = cached(function (_) {
          isCached = false;
          var html = '';
    
          if (!text) {
            return text
          }
    
          if (isPrimitive(text)) {
            html = compile(text);
          } else {
            html = compile.parser(text);
          }
    
          html = config.noEmoji ? html : emojify(html);
          slugify.clear();
    
          return html
        })(text);
    
        var curFileName = this$1.router.parse().file;
    
        if (isCached) {
          this$1.toc = this$1.cacheTOC[curFileName];
        } else {
          this$1.cacheTOC[curFileName] = [].concat( this$1.toc );
        }
    
        return result
      };
    };
    
    Compiler.prototype.compileEmbed = function compileEmbed (href, title) {
      var ref = getAndRemoveConfig(title);
        var str = ref.str;
        var config = ref.config;
      var embed;
      title = str;
    
      if (config.include) {
        if (!isAbsolutePath(href)) {
          href = getPath(
            this.contentBase,
            getParentPath(this.router.getCurrentPath()),
            href
          );
        }
    
        var media;
        if (config.type && (media = compileMedia[config.type])) {
          embed = media.call(this, href, title);
          embed.type = config.type;
        } else {
          var type = 'code';
          if (/\.(md|markdown)/.test(href)) {
            type = 'markdown';
          } else if (/\.mmd/.test(href)) {
            type = 'mermaid';
          } else if (/\.html?/.test(href)) {
            type = 'iframe';
          } else if (/\.(mp4|ogg)/.test(href)) {
            type = 'video';
          } else if (/\.mp3/.test(href)) {
            type = 'audio';
          }
          embed = compileMedia[type].call(this, href, title);
          embed.type = type;
        }
        embed.fragment = config.fragment;
    
        return embed
      }
    };
    
    Compiler.prototype._matchNotCompileLink = function _matchNotCompileLink (link) {
      var links = this.config.noCompileLinks || [];
    
      for (var i = 0; i < links.length; i++) {
        var n = links[i];
        var re = cachedLinks[n] || (cachedLinks[n] = new RegExp(("^" + n + "$")));
    
        if (re.test(link)) {
          return link
        }
      }
    };
    
    Compiler.prototype._initRenderer = function _initRenderer () {
      var renderer = new marked.Renderer();
      var ref = this;
        var linkTarget = ref.linkTarget;
        var router = ref.router;
        var contentBase = ref.contentBase;
      var _self = this;
      var origin = {};
    
      /**
       * Render anchor tag
       * @link https://github.com/markedjs/marked#overriding-renderer-methods
       */
      origin.heading = renderer.heading = function (text, level) {
        var ref = getAndRemoveConfig(text);
          var str = ref.str;
          var config = ref.config;
        var nextToc = {level: level, title: str};
    
        if (/{docsify-ignore}/g.test(str)) {
          str = str.replace('{docsify-ignore}', '');
          nextToc.title = str;
          nextToc.ignoreSubHeading = true;
        }
    
        if (/{docsify-ignore-all}/g.test(str)) {
          str = str.replace('{docsify-ignore-all}', '');
          nextToc.title = str;
          nextToc.ignoreAllSubs = true;
        }
    
        var slug = slugify(config.id || str);
        var url = router.toURL(router.getCurrentPath(), {id: slug});
        nextToc.slug = url;
        _self.toc.push(nextToc);
    
        return ("<h" + level + " id=\"" + slug + "\"><a href=\"" + url + "\" data-id=\"" + slug + "\" class=\"anchor\"><span>" + str + "</span></a></h" + level + ">")
      };
      // Highlight code
      origin.code = renderer.code = function (code, lang) {
          if ( lang === void 0 ) lang = '';
    
        code = code.replace(/@DOCSIFY_QM@/g, '`');
        var hl = prism.highlight(
          code,
          prism.languages[lang] || prism.languages.markup
        );
    
        return ("<pre v-pre data-lang=\"" + lang + "\"><code class=\"lang-" + lang + "\">" + hl + "</code></pre>")
      };
      origin.link = renderer.link = function (href, title, text) {
          if ( title === void 0 ) title = '';
    
        var attrs = '';
    
        var ref = getAndRemoveConfig(title);
          var str = ref.str;
          var config = ref.config;
        title = str;
    
        if (
          !isAbsolutePath(href) &&
          !_self._matchNotCompileLink(href) &&
          !config.ignore
        ) {
          if (href === _self.config.homepage) {
            href = 'README';
          }
          href = router.toURL(href, null, router.getCurrentPath());
        } else {
          attrs += href.indexOf('mailto:') === 0 ? '' : (" target=\"" + linkTarget + "\"");
        }
    
        if (config.target) {
          attrs += ' target=' + config.target;
        }
    
        if (config.disabled) {
          attrs += ' disabled';
          href = 'javascript:void(0)';
        }
    
        if (title) {
          attrs += " title=\"" + title + "\"";
        }
    
        return ("<a href=\"" + href + "\"" + attrs + ">" + text + "</a>")
      };
      origin.paragraph = renderer.paragraph = function (text) {
        var result;
        if (/^!&gt;/.test(text)) {
          result = helper('tip', text);
        } else if (/^\?&gt;/.test(text)) {
          result = helper('warn', text);
        } else {
          result = "<p>" + text + "</p>";
        }
        return result
      };
      origin.image = renderer.image = function (href, title, text) {
        var url = href;
        var attrs = '';
    
        var ref = getAndRemoveConfig(title);
          var str = ref.str;
          var config = ref.config;
        title = str;
    
        if (config['no-zoom']) {
          attrs += ' data-no-zoom';
        }
    
        if (title) {
          attrs += " title=\"" + title + "\"";
        }
    
        var size = config.size;
        if (size) {
          var sizes = size.split('x');
          if (sizes[1]) {
            attrs += 'width=' + sizes[0] + ' height=' + sizes[1];
          } else {
            attrs += 'width=' + sizes[0];
          }
        }
    
        if (!isAbsolutePath(href)) {
          url = getPath(contentBase, getParentPath(router.getCurrentPath()), href);
        }
    
        return ("<img src=\"" + url + "\"data-origin=\"" + href + "\" alt=\"" + text + "\"" + attrs + ">")
      };
      origin.list = renderer.list = function (body, ordered, start) {
        var isTaskList = /<li class="task-list-item">/.test(body.split('class="task-list"')[0]);
        var isStartReq = start && start > 1;
        var tag = ordered ? 'ol' : 'ul';
        var tagAttrs = [
          (isTaskList ? 'class="task-list"' : ''),
          (isStartReq ? ("start=\"" + start + "\"") : '')
        ].join(' ').trim();
    
        return ("<" + tag + " " + tagAttrs + ">" + body + "</" + tag + ">")
      };
      origin.listitem = renderer.listitem = function (text) {
        var isTaskItem = /^(<input.*type="checkbox"[^>]*>)/.test(text);
        var html = isTaskItem ? ("<li class=\"task-list-item\"><label>" + text + "</label></li>") : ("<li>" + text + "</li>");
    
        return html
      };
    
      renderer.origin = origin;
    
      return renderer
    };
    
    /**
     * Compile sidebar
     */
    Compiler.prototype.sidebar = function sidebar (text, level) {
      var ref = this;
        var toc = ref.toc;
      var currentPath = this.router.getCurrentPath();
      var html = '';
    
      if (text) {
        html = this.compile(text);
      } else {
        for (var i = 0; i < toc.length; i++) {
          if (toc[i].ignoreSubHeading) {
            var deletedHeaderLevel = toc[i].level;
            toc.splice(i, 1);
            // Remove headers who are under current header
            for (var j = i; deletedHeaderLevel < toc[j].level && j < toc.length; j++) {
              toc.splice(j, 1) && j-- && i++;
            }
            i--;
          }
        }
        var tree$$1 = this.cacheTree[currentPath] || genTree(toc, level);
        html = tree(tree$$1, '<ul>{inner}</ul>');
        this.cacheTree[currentPath] = tree$$1;
      }
    
      return html
    };
    
    /**
     * Compile sub sidebar
     */
    Compiler.prototype.subSidebar = function subSidebar (level) {
      if (!level) {
        this.toc = [];
        return
      }
      var currentPath = this.router.getCurrentPath();
      var ref = this;
        var cacheTree = ref.cacheTree;
        var toc = ref.toc;
    
      toc[0] && toc[0].ignoreAllSubs && toc.splice(0);
      toc[0] && toc[0].level === 1 && toc.shift();
    
      for (var i = 0; i < toc.length; i++) {
        toc[i].ignoreSubHeading && toc.splice(i, 1) && i--;
      }
      var tree$$1 = cacheTree[currentPath] || genTree(toc, level);
    
      cacheTree[currentPath] = tree$$1;
      this.toc = [];
      return tree(tree$$1)
    };
    
    Compiler.prototype.article = function article (text) {
      return this.compile(text)
    };
    
    /**
     * Compile cover page
     */
    Compiler.prototype.cover = function cover$$1 (text) {
      var cacheToc = this.toc.slice();
      var html = this.compile(text);
    
      this.toc = cacheToc.slice();
    
      return html
    };
    
    var title = $.title;
    /**
     * Toggle button
     */
    function btn(el) {
      var toggle = function (_) { return body.classList.toggle('close'); };
    
      el = getNode(el);
      if (el == null) {
        return
      }
      on(el, 'click', function (e) {
        e.stopPropagation();
        toggle();
      });
    
      isMobile &&
        on(
          body,
          'click',
          function (_) { return body.classList.contains('close') && toggle(); }
        );
    }
    
    function collapse(el) {
      el = getNode(el);
      if (el == null) {
        return
      }
      on(el, 'click', function (ref) {
        var target = ref.target;
    
        if (
          target.nodeName === 'A' &&
          target.nextSibling &&
          target.nextSibling.classList.contains('app-sub-sidebar')
        ) {
          toggleClass(target.parentNode, 'collapse');
        }
      });
    }
    
    function sticky() {
      var cover = getNode('section.cover');
      if (!cover) {
        return
      }
      var coverHeight = cover.getBoundingClientRect().height;
    
      if (window.pageYOffset >= coverHeight || cover.classList.contains('hidden')) {
        toggleClass(body, 'add', 'sticky');
      } else {
        toggleClass(body, 'remove', 'sticky');
      }
    }
    
    /**
     * Get and active link
     * @param  {object} router
     * @param  {string|element}  el
     * @param  {Boolean} isParent   acitve parent
     * @param  {Boolean} autoTitle  auto set title
     * @return {element}
     */
    function getAndActive(router, el, isParent, autoTitle) {
      el = getNode(el);
      var links = [];
      if (el != null) {
        links = findAll(el, 'a');
      }
      var hash = decodeURI(router.toURL(router.getCurrentPath()));
      var target;
    
      links.sort(function (a, b) { return b.href.length - a.href.length; }).forEach(function (a) {
        var href = a.getAttribute('href');
        var node = isParent ? a.parentNode : a;
    
        if (hash.indexOf(href) === 0 && !target) {
          target = a;
          toggleClass(node, 'add', 'active');
        } else {
          toggleClass(node, 'remove', 'active');
        }
      });
    
      if (autoTitle) {
        $.title = target ? (target.title || ((target.innerText) + " - " + title)) : title;
      }
    
      return target
    }
    
    var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) { descriptor.writable = true; } Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) { defineProperties(Constructor.prototype, protoProps); } if (staticProps) { defineProperties(Constructor, staticProps); } return Constructor; }; }();
    
    function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
    
    var Tweezer = function () {
      function Tweezer() {
        var opts = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
    
        _classCallCheck(this, Tweezer);
    
        this.duration = opts.duration || 1000;
        this.ease = opts.easing || this._defaultEase;
        this.start = opts.start;
        this.end = opts.end;
    
        this.frame = null;
        this.next = null;
        this.isRunning = false;
        this.events = {};
        this.direction = this.start < this.end ? 'up' : 'down';
      }
    
      _createClass(Tweezer, [{
        key: 'begin',
        value: function begin() {
          if (!this.isRunning && this.next !== this.end) {
            this.frame = window.requestAnimationFrame(this._tick.bind(this));
          }
          return this;
        }
      }, {
        key: 'stop',
        value: function stop() {
          window.cancelAnimationFrame(this.frame);
          this.isRunning = false;
          this.frame = null;
          this.timeStart = null;
          this.next = null;
          return this;
        }
      }, {
        key: 'on',
        value: function on(name, handler) {
          this.events[name] = this.events[name] || [];
          this.events[name].push(handler);
          return this;
        }
      }, {
        key: 'emit',
        value: function emit(name, val) {
          var _this = this;
    
          var e = this.events[name];
          e && e.forEach(function (handler) {
            return handler.call(_this, val);
          });
        }
      }, {
        key: '_tick',
        value: function _tick(currentTime) {
          this.isRunning = true;
    
          var lastTick = this.next || this.start;
    
          if (!this.timeStart) { this.timeStart = currentTime; }
          this.timeElapsed = currentTime - this.timeStart;
          this.next = Math.round(this.ease(this.timeElapsed, this.start, this.end - this.start, this.duration));
    
          if (this._shouldTick(lastTick)) {
            this.emit('tick', this.next);
            this.frame = window.requestAnimationFrame(this._tick.bind(this));
          } else {
            this.emit('tick', this.end);
            this.emit('done', null);
          }
        }
      }, {
        key: '_shouldTick',
        value: function _shouldTick(lastTick) {
          return {
            up: this.next < this.end && lastTick <= this.next,
            down: this.next > this.end && lastTick >= this.next
          }[this.direction];
        }
      }, {
        key: '_defaultEase',
        value: function _defaultEase(t, b, c, d) {
          if ((t /= d / 2) < 1) { return c / 2 * t * t + b; }
          return -c / 2 * (--t * (t - 2) - 1) + b;
        }
      }]);
    
      return Tweezer;
    }();
    
    var nav = {};
    var hoverOver = false;
    var scroller = null;
    var enableScrollEvent = true;
    var coverHeight = 0;
    
    function scrollTo(el) {
      if (scroller) {
        scroller.stop();
      }
      enableScrollEvent = false;
      scroller = new Tweezer({
        start: window.pageYOffset,
        end: el.getBoundingClientRect().top + window.pageYOffset,
        duration: 500
      })
        .on('tick', function (v) { return window.scrollTo(0, v); })
        .on('done', function () {
          enableScrollEvent = true;
          scroller = null;
        })
        .begin();
    }
    
    function highlight(path) {
      if (!enableScrollEvent) {
        return
      }
      var sidebar = getNode('.sidebar');
      var anchors = findAll('.anchor');
      var wrap = find(sidebar, '.sidebar-nav');
      var active = find(sidebar, 'li.active');
      var doc = document.documentElement;
      var top = ((doc && doc.scrollTop) || document.body.scrollTop) - coverHeight;
      var last;
    
      for (var i = 0, len = anchors.length; i < len; i += 1) {
        var node = anchors[i];
    
        if (node.offsetTop > top) {
          if (!last) {
            last = node;
          }
          break
        } else {
          last = node;
        }
      }
      if (!last) {
        return
      }
      var li = nav[getNavKey(decodeURIComponent(path), last.getAttribute('data-id'))];
    
      if (!li || li === active) {
        return
      }
    
      active && active.classList.remove('active');
      li.classList.add('active');
      active = li;
    
      // Scroll into view
      // https://github.com/vuejs/vuejs.org/blob/master/themes/vue/source/js/common.js#L282-L297
      if (!hoverOver && body.classList.contains('sticky')) {
        var height = sidebar.clientHeight;
        var curOffset = 0;
        var cur = active.offsetTop + active.clientHeight + 40;
        var isInView =
          active.offsetTop >= wrap.scrollTop && cur <= wrap.scrollTop + height;
        var notThan = cur - curOffset < height;
        var top$1 = isInView ? wrap.scrollTop : notThan ? curOffset : cur - height;
    
        sidebar.scrollTop = top$1;
      }
    }
    
    function getNavKey(path, id) {
      return (path + "?id=" + id)
    }
    
    function scrollActiveSidebar(router) {
      var cover = find('.cover.show');
      coverHeight = cover ? cover.offsetHeight : 0;
    
      var sidebar = getNode('.sidebar');
      var lis = [];
      if (sidebar != null) {
        lis = findAll(sidebar, 'li');
      }
    
      for (var i = 0, len = lis.length; i < len; i += 1) {
        var li = lis[i];
        var a = li.querySelector('a');
        if (!a) {
          continue
        }
        var href = a.getAttribute('href');
    
        if (href !== '/') {
          var ref = router.parse(href);
          var id = ref.query.id;
          var path$1 = ref.path;
          if (id) {
            href = getNavKey(path$1, id);
          }
        }
    
        if (href) {
          nav[decodeURIComponent(href)] = li;
        }
      }
    
      if (isMobile) {
        return
      }
      var path = router.getCurrentPath();
      off('scroll', function () { return highlight(path); });
      on('scroll', function () { return highlight(path); });
      on(sidebar, 'mouseover', function () {
        hoverOver = true;
      });
      on(sidebar, 'mouseleave', function () {
        hoverOver = false;
      });
    }
    
    function scrollIntoView(path, id) {
      if (!id) {
        return
      }
    
      var section = find('#' + id);
      section && scrollTo(section);
    
      var li = nav[getNavKey(path, id)];
      var sidebar = getNode('.sidebar');
      var active = find(sidebar, 'li.active');
      active && active.classList.remove('active');
      li && li.classList.add('active');
    }
    
    var scrollEl = $.scrollingElement || $.documentElement;
    
    function scroll2Top(offset) {
      if ( offset === void 0 ) offset = 0;
    
      scrollEl.scrollTop = offset === true ? 0 : Number(offset);
    }
    
    var cached$1 = {};
    
    function walkFetchEmbed(ref, cb) {
      var embedTokens = ref.embedTokens;
      var compile = ref.compile;
      var fetch = ref.fetch;
    
      var token;
      var step = 0;
      var count = 1;
    
      if (!embedTokens.length) {
        return cb({})
      }
    
      while ((token = embedTokens[step++])) {
        var next = (function (token) {
          return function (text) {
            var embedToken;
            if (text) {
              if (token.embed.type === 'markdown') {
                embedToken = compile.lexer(text);
              } else if (token.embed.type === 'code') {
                if (token.embed.fragment) {
                  var fragment = token.embed.fragment;
                  var pattern = new RegExp(("(?:###|\\/\\/\\/)\\s*\\[" + fragment + "\\]([\\s\\S]*)(?:###|\\/\\/\\/)\\s*\\[" + fragment + "\\]"));
                  text = ((text.match(pattern) || [])[1] || '').trim();
                }
                embedToken = compile.lexer(
                  '```' +
                    token.embed.lang +
                    '\n' +
                    text.replace(/`/g, '@DOCSIFY_QM@') +
                    '\n```\n'
                );
              } else if (token.embed.type === 'mermaid') {
                embedToken = [
                  {type: 'html', text: ("<div class=\"mermaid\">\n" + text + "\n</div>")}
                ];
                embedToken.links = {};
              } else {
                embedToken = [{type: 'html', text: text}];
                embedToken.links = {};
              }
            }
            cb({token: token, embedToken: embedToken});
            if (++count >= step) {
              cb({});
            }
          }
        })(token);
    
        if (token.embed.url) {
          {
            get(token.embed.url).then(next);
          }
        } else {
          next(token.embed.html);
        }
      }
    }
    
    function prerenderEmbed(ref, done) {
      var compiler = ref.compiler;
      var raw = ref.raw; if ( raw === void 0 ) raw = '';
      var fetch = ref.fetch;
    
      var hit = cached$1[raw];
      if (hit) {
        var copy = hit.slice();
        copy.links = hit.links;
        return done(copy)
      }
    
      var compile = compiler._marked;
      var tokens = compile.lexer(raw);
      var embedTokens = [];
      var linkRE = compile.InlineLexer.rules.link;
      var links = tokens.links;
    
      tokens.forEach(function (token, index) {
        if (token.type === 'paragraph') {
          token.text = token.text.replace(
            new RegExp(linkRE.source, 'g'),
            function (src, filename, href, title) {
              var embed = compiler.compileEmbed(href, title);
    
              if (embed) {
                embedTokens.push({
                  index: index,
                  embed: embed
                });
              }
    
              return src
            }
          );
        }
      });
    
      var moveIndex = 0;
      walkFetchEmbed({compile: compile, embedTokens: embedTokens, fetch: fetch}, function (ref) {
        var embedToken = ref.embedToken;
        var token = ref.token;
    
        if (token) {
          var index = token.index + moveIndex;
    
          merge(links, embedToken.links);
    
          tokens = tokens
            .slice(0, index)
            .concat(embedToken, tokens.slice(index + 1));
          moveIndex += embedToken.length - 1;
        } else {
          cached$1[raw] = tokens.concat();
          tokens.links = cached$1[raw].links = links;
          done(tokens);
        }
      });
    }
    
    function executeScript() {
      var script = findAll('.markdown-section>script')
        .filter(function (s) { return !/template/.test(s.type); })[0];
      if (!script) {
        return false
      }
      var code = script.innerText.trim();
      if (!code) {
        return false
      }
    
      setTimeout(function (_) {
        window.__EXECUTE_RESULT__ = new Function(code)();
      }, 0);
    }
    
    function formatUpdated(html, updated, fn) {
      updated =
        typeof fn === 'function' ?
          fn(updated) :
          typeof fn === 'string' ?
            tinydate(fn)(new Date(updated)) :
            updated;
    
      return html.replace(/{docsify-updated}/g, updated)
    }
    
    function renderMain(html) {
      if (!html) {
        html = '<h1>404 - Not found</h1>';
      }
    
      this._renderTo('.markdown-section', html);
      // Render sidebar with the TOC
      !this.config.loadSidebar && this._renderSidebar();
    
      // Execute script
      if (
        this.config.executeScript !== false &&
        typeof window.Vue !== 'undefined' &&
        !executeScript()
      ) {
        setTimeout(function (_) {
          var vueVM = window.__EXECUTE_RESULT__;
          vueVM && vueVM.$destroy && vueVM.$destroy();
          window.__EXECUTE_RESULT__ = new window.Vue().$mount('#main');
        }, 0);
      } else {
        this.config.executeScript && executeScript();
      }
    }
    
    function renderNameLink(vm) {
      var el = getNode('.app-name-link');
      var nameLink = vm.config.nameLink;
      var path = vm.route.path;
    
      if (!el) {
        return
      }
    
      if (isPrimitive(vm.config.nameLink)) {
        el.setAttribute('href', nameLink);
      } else if (typeof nameLink === 'object') {
        var match = Object.keys(nameLink).filter(function (key) { return path.indexOf(key) > -1; })[0];
    
        el.setAttribute('href', nameLink[match]);
      }
    }
    
    function renderMixin(proto) {
      proto._renderTo = function (el, content, replace) {
        var node = getNode(el);
        if (node) {
          node[replace ? 'outerHTML' : 'innerHTML'] = content;
        }
      };
    
      proto._renderSidebar = function (text) {
        var ref = this.config;
        var maxLevel = ref.maxLevel;
        var subMaxLevel = ref.subMaxLevel;
        var loadSidebar = ref.loadSidebar;
    
        this._renderTo('.sidebar-nav', this.compiler.sidebar(text, maxLevel));
        var activeEl = getAndActive(this.router, '.sidebar-nav', true, true);
        if (loadSidebar && activeEl) {
          activeEl.parentNode.innerHTML +=
            this.compiler.subSidebar(subMaxLevel) || '';
        } else {
          // Reset toc
          this.compiler.subSidebar();
        }
        // Bind event
        this._bindEventOnRendered(activeEl);
      };
    
      proto._bindEventOnRendered = function (activeEl) {
        var ref = this.config;
        var autoHeader = ref.autoHeader;
        var auto2top = ref.auto2top;
    
        scrollActiveSidebar(this.router);
    
        if (autoHeader && activeEl) {
          var main$$1 = getNode('#main');
          var firstNode = main$$1.children[0];
          if (firstNode && firstNode.tagName !== 'H1') {
            var h1 = create('h1');
            h1.innerText = activeEl.innerText;
            before(main$$1, h1);
          }
        }
    
        auto2top && scroll2Top(auto2top);
      };
    
      proto._renderNav = function (text) {
        text && this._renderTo('nav', this.compiler.compile(text));
        if (this.config.loadNavbar) {
          getAndActive(this.router, 'nav');
        }
      };
    
      proto._renderMain = function (text, opt, next) {
        var this$1 = this;
        if ( opt === void 0 ) opt = {};
    
        if (!text) {
          return renderMain.call(this, text)
        }
    
        callHook(this, 'beforeEach', text, function (result) {
          var html;
          var callback = function () {
            if (opt.updatedAt) {
              html = formatUpdated(html, opt.updatedAt, this$1.config.formatUpdated);
            }
    
            callHook(this$1, 'afterEach', html, function (text) { return renderMain.call(this$1, text); });
          };
          if (this$1.isHTML) {
            html = this$1.result = text;
            callback();
            next();
          } else {
            prerenderEmbed(
              {
                compiler: this$1.compiler,
                raw: result
              },
              function (tokens) {
                html = this$1.compiler.compile(tokens);
                callback();
                next();
              }
            );
          }
        });
      };
    
      proto._renderCover = function (text, coverOnly) {
        var el = getNode('.cover');
    
        toggleClass(getNode('main'), coverOnly ? 'add' : 'remove', 'hidden');
        if (!text) {
          toggleClass(el, 'remove', 'show');
          return
        }
        toggleClass(el, 'add', 'show');
    
        var html = this.coverIsHTML ? text : this.compiler.cover(text);
    
        var m = html
          .trim()
          .match('<p><img.*?data-origin="(.*?)"[^a]+alt="(.*?)">([^<]*?)</p>$');
    
        if (m) {
          if (m[2] === 'color') {
            el.style.background = m[1] + (m[3] || '');
          } else {
            var path = m[1];
    
            toggleClass(el, 'add', 'has-mask');
            if (!isAbsolutePath(m[1])) {
              path = getPath(this.router.getBasePath(), m[1]);
            }
            el.style.backgroundImage = "url(" + path + ")";
            el.style.backgroundSize = 'cover';
            el.style.backgroundPosition = 'center center';
          }
          html = html.replace(m[0], '');
        }
    
        this._renderTo('.cover-main', html);
        sticky();
      };
    
      proto._updateRender = function () {
        // Render name link
        renderNameLink(this);
      };
    }
    
    function initRender(vm) {
      var config = vm.config;
    
      // Init markdown compiler
      vm.compiler = new Compiler(config, vm.router);
      if (inBrowser) {
        window.__current_docsify_compiler__ = vm.compiler;
      }
    
      var id = config.el || '#app';
      var navEl = find('nav') || create('nav');
    
      var el = find(id);
      var html = '';
      var navAppendToTarget = body;
    
      if (el) {
        if (config.repo) {
          html += corner(config.repo);
        }
        if (config.coverpage) {
          html += cover();
        }
    
        if (config.logo) {
          var isBase64 = /^data:image/.test(config.logo);
          var isExternal = /(?:http[s]?:)?\/\//.test(config.logo);
          var isRelative = /^\./.test(config.logo);
    
          if (!isBase64 && !isExternal && !isRelative) {
            config.logo = getPath(vm.router.getBasePath(), config.logo);
          }
        }
    
        html += main(config);
        // Render main app
        vm._renderTo(el, html, true);
      } else {
        vm.rendered = true;
      }
    
      if (config.mergeNavbar && isMobile) {
        navAppendToTarget = find('.sidebar');
      } else {
        navEl.classList.add('app-nav');
    
        if (!config.repo) {
          navEl.classList.add('no-badge');
        }
      }
    
      // Add nav
      if (config.loadNavbar) {
        before(navAppendToTarget, navEl);
      }
    
      if (config.themeColor) {
        $.head.appendChild(
          create('div', theme(config.themeColor)).firstElementChild
        );
        // Polyfll
        cssVars(config.themeColor);
      }
      vm._updateRender();
      toggleClass(body, 'ready');
    }
    
    var cached$2 = {};
    
    function getAlias(path, alias, last) {
      var match = Object.keys(alias).filter(function (key) {
        var re = cached$2[key] || (cached$2[key] = new RegExp(("^" + key + "$")));
        return re.test(path) && path !== last
      })[0];
    
      return match ?
        getAlias(path.replace(cached$2[match], alias[match]), alias, path) :
        path
    }
    
    function getFileName(path, ext) {
      return new RegExp(("\\.(" + (ext.replace(/^\./, '')) + "|html)$"), 'g').test(path) ?
        path :
        /\/$/g.test(path) ? (path + "README" + ext) : ("" + path + ext)
    }
    
    var History = function History(config) {
      this.config = config;
    };
    
    History.prototype.getBasePath = function getBasePath () {
      return this.config.basePath
    };
    
    History.prototype.getFile = function getFile (path, isRelative) {
        if ( path === void 0 ) path = this.getCurrentPath();
    
      var ref = this;
        var config = ref.config;
      var base = this.getBasePath();
      var ext = typeof config.ext === 'string' ? config.ext : '.md';
    
      path = config.alias ? getAlias(path, config.alias) : path;
      path = getFileName(path, ext);
      path = path === ("/README" + ext) ? config.homepage || path : path;
      path = isAbsolutePath(path) ? path : getPath(base, path);
    
      if (isRelative) {
        path = path.replace(new RegExp(("^" + base)), '');
      }
    
      return path
    };
    
    History.prototype.onchange = function onchange (cb) {
        if ( cb === void 0 ) cb = noop;
    
      cb();
    };
    
    History.prototype.getCurrentPath = function getCurrentPath () {};
    
    History.prototype.normalize = function normalize () {};
    
    History.prototype.parse = function parse () {};
    
    History.prototype.toURL = function toURL (path, params, currentRoute) {
      var local = currentRoute && path[0] === '#';
      var route = this.parse(replaceSlug(path));
    
      route.query = merge({}, route.query, params);
      path = route.path + stringifyQuery(route.query);
      path = path.replace(/\.md(\?)|\.md$/, '$1');
    
      if (local) {
        var idIndex = currentRoute.indexOf('?');
        path =
          (idIndex > 0 ? currentRoute.substring(0, idIndex) : currentRoute) + path;
      }
    
      if (this.config.relativePath && path.indexOf('/') !== 0) {
        var currentDir = currentRoute.substring(0, currentRoute.lastIndexOf('/') + 1);
        return cleanPath(resolvePath(currentDir + path))
      }
      return cleanPath('/' + path)
    };
    
    function replaceHash(path) {
      var i = location.href.indexOf('#');
      location.replace(location.href.slice(0, i >= 0 ? i : 0) + '#' + path);
    }
    
    var HashHistory = (function (History$$1) {
      function HashHistory(config) {
        History$$1.call(this, config);
        this.mode = 'hash';
      }
    
      if ( History$$1 ) HashHistory.__proto__ = History$$1;
      HashHistory.prototype = Object.create( History$$1 && History$$1.prototype );
      HashHistory.prototype.constructor = HashHistory;
    
      HashHistory.prototype.getBasePath = function getBasePath () {
        var path = window.location.pathname || '';
        var base = this.config.basePath;
    
        return /^(\/|https?:)/g.test(base) ? base : cleanPath(path + '/' + base)
      };
    
      HashHistory.prototype.getCurrentPath = function getCurrentPath () {
        // We can't use location.hash here because it's not
        // consistent across browsers - Firefox will pre-decode it!
        var href = location.href;
        var index = href.indexOf('#');
        return index === -1 ? '' : href.slice(index + 1)
      };
    
      HashHistory.prototype.onchange = function onchange (cb) {
        if ( cb === void 0 ) cb = noop;
    
        on('hashchange', cb);
      };
    
      HashHistory.prototype.normalize = function normalize () {
        var path = this.getCurrentPath();
    
        path = replaceSlug(path);
    
        if (path.charAt(0) === '/') {
          return replaceHash(path)
        }
        replaceHash('/' + path);
      };
    
      /**
       * Parse the url
       * @param {string} [path=location.herf]
       * @return {object} { path, query }
       */
      HashHistory.prototype.parse = function parse (path) {
        if ( path === void 0 ) path = location.href;
    
        var query = '';
    
        var hashIndex = path.indexOf('#');
        if (hashIndex >= 0) {
          path = path.slice(hashIndex + 1);
        }
    
        var queryIndex = path.indexOf('?');
        if (queryIndex >= 0) {
          query = path.slice(queryIndex + 1);
          path = path.slice(0, queryIndex);
        }
    
        return {
          path: path,
          file: this.getFile(path, true),
          query: parseQuery(query)
        }
      };
    
      HashHistory.prototype.toURL = function toURL (path, params, currentRoute) {
        return '#' + History$$1.prototype.toURL.call(this, path, params, currentRoute)
      };
    
      return HashHistory;
    }(History));
    
    var HTML5History = (function (History$$1) {
      function HTML5History(config) {
        History$$1.call(this, config);
        this.mode = 'history';
      }
    
      if ( History$$1 ) HTML5History.__proto__ = History$$1;
      HTML5History.prototype = Object.create( History$$1 && History$$1.prototype );
      HTML5History.prototype.constructor = HTML5History;
    
      HTML5History.prototype.getCurrentPath = function getCurrentPath () {
        var base = this.getBasePath();
        var path = window.location.pathname;
    
        if (base && path.indexOf(base) === 0) {
          path = path.slice(base.length);
        }
    
        return (path || '/') + window.location.search + window.location.hash
      };
    
      HTML5History.prototype.onchange = function onchange (cb) {
        if ( cb === void 0 ) cb = noop;
    
        on('click', function (e) {
          var el = e.target.tagName === 'A' ? e.target : e.target.parentNode;
    
          if (el.tagName === 'A' && !/_blank/.test(el.target)) {
            e.preventDefault();
            var url = el.href;
            window.history.pushState({key: url}, '', url);
            cb();
          }
        });
    
        on('popstate', cb);
      };
    
      /**
       * Parse the url
       * @param {string} [path=location.href]
       * @return {object} { path, query }
       */
      HTML5History.prototype.parse = function parse (path) {
        if ( path === void 0 ) path = location.href;
    
        var query = '';
    
        var queryIndex = path.indexOf('?');
        if (queryIndex >= 0) {
          query = path.slice(queryIndex + 1);
          path = path.slice(0, queryIndex);
        }
    
        var base = getPath(location.origin);
        var baseIndex = path.indexOf(base);
    
        if (baseIndex > -1) {
          path = path.slice(baseIndex + base.length);
        }
    
        return {
          path: path,
          file: this.getFile(path),
          query: parseQuery(query)
        }
      };
    
      return HTML5History;
    }(History));
    
    function routerMixin(proto) {
      proto.route = {};
    }
    
    var lastRoute = {};
    
    function updateRender(vm) {
      vm.router.normalize();
      vm.route = vm.router.parse();
      body.setAttribute('data-page', vm.route.file);
    }
    
    function initRouter(vm) {
      var config = vm.config;
      var mode = config.routerMode || 'hash';
      var router;
    
      if (mode === 'history' && supportsPushState) {
        router = new HTML5History(config);
      } else {
        router = new HashHistory(config);
      }
    
      vm.router = router;
      updateRender(vm);
      lastRoute = vm.route;
    
      router.onchange(function (_) {
        updateRender(vm);
        vm._updateRender();
    
        if (lastRoute.path === vm.route.path) {
          vm.$resetEvents();
          return
        }
    
        vm.$fetch();
        lastRoute = vm.route;
      });
    }
    
    function eventMixin(proto) {
      proto.$resetEvents = function () {
        scrollIntoView(this.route.path, this.route.query.id);
    
        if (this.config.loadNavbar) {
          getAndActive(this.router, 'nav');
        }
      };
    }
    
    function initEvent(vm) {
      // Bind toggle button
      btn('button.sidebar-toggle', vm.router);
      collapse('.sidebar', vm.router);
      // Bind sticky effect
      if (vm.config.coverpage) {
        !isMobile && on('scroll', sticky);
      } else {
        body.classList.add('sticky');
      }
    }
    
    function loadNested(path, qs, file, next, vm, first) {
      path = first ? path : path.replace(/\/$/, '');
      path = getParentPath(path);
    
      if (!path) {
        return
      }
    
      get(
        vm.router.getFile(path + file) + qs,
        false,
        vm.config.requestHeaders
      ).then(next, function (_) { return loadNested(path, qs, file, next, vm); });
    }
    
    function fetchMixin(proto) {
      var last;
    
      var abort = function () { return last && last.abort && last.abort(); };
      var request = function (url, hasbar, requestHeaders) {
        abort();
        last = get(url, true, requestHeaders);
        return last
      };
    
      var get404Path = function (path, config) {
        var notFoundPage = config.notFoundPage;
        var ext = config.ext;
        var defaultPath = '_404' + (ext || '.md');
        var key;
        var path404;
    
        switch (typeof notFoundPage) {
          case 'boolean':
            path404 = defaultPath;
            break
          case 'string':
            path404 = notFoundPage;
            break
    
          case 'object':
            key = Object.keys(notFoundPage)
              .sort(function (a, b) { return b.length - a.length; })
              .find(function (key) { return path.match(new RegExp('^' + key)); });
    
            path404 = (key && notFoundPage[key]) || defaultPath;
            break
    
          default:
            break
        }
    
        return path404
      };
    
      proto._loadSideAndNav = function (path, qs, loadSidebar, cb) {
        var this$1 = this;
    
        return function () {
          if (!loadSidebar) {
            return cb()
          }
    
          var fn = function (result) {
            this$1._renderSidebar(result);
            cb();
          };
    
          // Load sidebar
          loadNested(path, qs, loadSidebar, fn, this$1, true);
        }
      };
    
      proto._fetch = function (cb) {
        var this$1 = this;
        if ( cb === void 0 ) cb = noop;
    
        var ref = this.route;
        var path = ref.path;
        var query = ref.query;
        var qs = stringifyQuery(query, ['id']);
        var ref$1 = this.config;
        var loadNavbar = ref$1.loadNavbar;
        var requestHeaders = ref$1.requestHeaders;
        var loadSidebar = ref$1.loadSidebar;
        // Abort last request
    
        var file = this.router.getFile(path);
        var req = request(file + qs, true, requestHeaders);
    
        // Current page is html
        this.isHTML = /\.html$/g.test(file);
    
        // Load main content
        req.then(
          function (text, opt) { return this$1._renderMain(
              text,
              opt,
              this$1._loadSideAndNav(path, qs, loadSidebar, cb)
            ); },
          function (_) {
            this$1._fetchFallbackPage(file, qs, cb) || this$1._fetch404(file, qs, cb);
          }
        );
    
        // Load nav
        loadNavbar &&
          loadNested(
            path,
            qs,
            loadNavbar,
            function (text) { return this$1._renderNav(text); },
            this,
            true
          );
      };
    
      proto._fetchCover = function () {
        var this$1 = this;
    
        var ref = this.config;
        var coverpage = ref.coverpage;
        var requestHeaders = ref.requestHeaders;
        var query = this.route.query;
        var root = getParentPath(this.route.path);
    
        if (coverpage) {
          var path = null;
          var routePath = this.route.path;
          if (typeof coverpage === 'string') {
            if (routePath === '/') {
              path = coverpage;
            }
          } else if (Array.isArray(coverpage)) {
            path = coverpage.indexOf(routePath) > -1 && '_coverpage';
          } else {
            var cover = coverpage[routePath];
            path = cover === true ? '_coverpage' : cover;
          }
    
          var coverOnly = Boolean(path) && this.config.onlyCover;
          if (path) {
            path = this.router.getFile(root + path);
            this.coverIsHTML = /\.html$/g.test(path);
            get(path + stringifyQuery(query, ['id']), false, requestHeaders).then(
              function (text) { return this$1._renderCover(text, coverOnly); }
            );
          } else {
            this._renderCover(null, coverOnly);
          }
          return coverOnly
        }
      };
    
      proto.$fetch = function (cb) {
        var this$1 = this;
        if ( cb === void 0 ) cb = noop;
    
        var done = function () {
          callHook(this$1, 'doneEach');
          cb();
        };
    
        var onlyCover = this._fetchCover();
    
        if (onlyCover) {
          done();
        } else {
          this._fetch(function () {
            this$1.$resetEvents();
            done();
          });
        }
      };
    
      proto._fetchFallbackPage = function (path, qs, cb) {
        var this$1 = this;
        if ( cb === void 0 ) cb = noop;
    
        var ref = this.config;
        var requestHeaders = ref.requestHeaders;
        var fallbackLanguages = ref.fallbackLanguages;
        var loadSidebar = ref.loadSidebar;
    
        if (!fallbackLanguages) {
          return false
        }
    
        var local = path.split('/')[1];
    
        if (fallbackLanguages.indexOf(local) === -1) {
          return false
        }
        var newPath = path.replace(new RegExp(("^/" + local)), '');
        var req = request(newPath + qs, true, requestHeaders);
    
        req.then(
          function (text, opt) { return this$1._renderMain(
              text,
              opt,
              this$1._loadSideAndNav(path, qs, loadSidebar, cb)
            ); },
          function () { return this$1._fetch404(path, qs, cb); }
        );
    
        return true
      };
      /**
       * Load the 404 page
       * @param path
       * @param qs
       * @param cb
       * @returns {*}
       * @private
       */
      proto._fetch404 = function (path, qs, cb) {
        var this$1 = this;
        if ( cb === void 0 ) cb = noop;
    
        var ref = this.config;
        var loadSidebar = ref.loadSidebar;
        var requestHeaders = ref.requestHeaders;
        var notFoundPage = ref.notFoundPage;
    
        var fnLoadSideAndNav = this._loadSideAndNav(path, qs, loadSidebar, cb);
        if (notFoundPage) {
          var path404 = get404Path(path, this.config);
    
          request(this.router.getFile(path404), true, requestHeaders).then(
            function (text, opt) { return this$1._renderMain(text, opt, fnLoadSideAndNav); },
            function () { return this$1._renderMain(null, {}, fnLoadSideAndNav); }
          );
          return true
        }
    
        this._renderMain(null, {}, fnLoadSideAndNav);
        return false
      };
    }
    
    function initFetch(vm) {
      var ref = vm.config;
      var loadSidebar = ref.loadSidebar;
    
      // Server-Side Rendering
      if (vm.rendered) {
        var activeEl = getAndActive(vm.router, '.sidebar-nav', true, true);
        if (loadSidebar && activeEl) {
          activeEl.parentNode.innerHTML += window.__SUB_SIDEBAR__;
        }
        vm._bindEventOnRendered(activeEl);
        vm.$resetEvents();
        callHook(vm, 'doneEach');
        callHook(vm, 'ready');
      } else {
        vm.$fetch(function (_) { return callHook(vm, 'ready'); });
      }
    }
    
    function initMixin(proto) {
      proto._init = function () {
        var vm = this;
        vm.config = config();
    
        initLifecycle(vm); // Init hooks
        initPlugin(vm); // Install plugins
        callHook(vm, 'init');
        initRouter(vm); // Add router
        initRender(vm); // Render base DOM
        initEvent(vm); // Bind events
        initFetch(vm); // Fetch data
        callHook(vm, 'mounted');
      };
    }
    
    function initPlugin(vm) {
      [].concat(vm.config.plugins).forEach(function (fn) { return isFn(fn) && fn(vm._lifecycle, vm); });
    }
    
    
    
    var util = Object.freeze({
        cached: cached,
        hyphenate: hyphenate,
        hasOwn: hasOwn,
        merge: merge,
        isPrimitive: isPrimitive,
        noop: noop,
        isFn: isFn,
        inBrowser: inBrowser,
        isMobile: isMobile,
        supportsPushState: supportsPushState,
        parseQuery: parseQuery,
        stringifyQuery: stringifyQuery,
        isAbsolutePath: isAbsolutePath,
        getParentPath: getParentPath,
        cleanPath: cleanPath,
        resolvePath: resolvePath,
        getPath: getPath,
        replaceSlug: replaceSlug
    });
    
    function initGlobalAPI () {
      window.Docsify = {
        util: util,
        dom: dom,
        get: get,
        slugify: slugify,
        version: '4.9.4'
      };
      window.DocsifyCompiler = Compiler;
      window.marked = marked;
      window.Prism = prism;
    }
    
    /**
     * Fork https://github.com/bendrucker/document-ready/blob/master/index.js
     */
    function ready(callback) {
      var state = document.readyState;
    
      if (state === 'complete' || state === 'interactive') {
        return setTimeout(callback, 0)
      }
    
      document.addEventListener('DOMContentLoaded', callback);
    }
    
    function Docsify() {
      this._init();
    }
    
    var proto = Docsify.prototype;
    
    initMixin(proto);
    routerMixin(proto);
    renderMixin(proto);
    fetchMixin(proto);
    eventMixin(proto);
    
    /**
     * Global API
     */
    initGlobalAPI();
    
    /**
     * Run Docsify
     */
    ready(function (_) { return new Docsify(); });
    
    }());
    